#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <omp.h>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static constexpr const char * kSourcePath =
    "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf";
static constexpr const char * kSourceSha =
    "e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d";
static constexpr const char * kOracleAbi = "escha-reconstruct-cba-h128-fp32-v1";
static constexpr const char * kQuantAbi = "ggml-kquants-numpy-v1-batch4096";
static constexpr const char * kNativeAbi = "exp11-attempt2-native-slice1-v1";
static constexpr uint32_t kQ2K = 10;
static constexpr size_t kQK = 256;
static constexpr size_t kQ2Bytes = 84;

struct TensorSpec {
    const char * role;
    const char * output_name;
    uint64_t code_offset;
    uint64_t code_bytes;
    uint64_t rin_offset;
    uint64_t rout_offset;
    int ic;
    int oc;
    int K;
    const char * expected_payload_sha;
};

static constexpr std::array<TensorSpec, 3> kSpecs{{
    {"ffn_gate", "blk.0.ffn_gate.weight", 41399744, 22282240, 63681984, 63692224,
     5120, 17408, 2, "ea4cb733b800d50e7be74a2ff84f22fa8eed79b32631ee73d0117213d2305a40"},
    {"ffn_up", "blk.0.ffn_up.weight", 63796672, 33423360, 97220032, 97230272,
     5120, 17408, 3, "041c0045017d17bdfb04698a36a5311032e04e309e8a95872fbadcc691b6a713"},
    {"ffn_down", "blk.0.ffn_down.weight", 97334720, 33423360, 130758080, 130792896,
     17408, 5120, 3, "e5a94a62c8c4e9f9a7450007ab0fb5b5070540a63a939285b4450f2dd40d28b7"},
}};

static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static std::string json_escape(std::string_view s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"': o << "\\\""; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
                else o << char(c);
        }
    }
    return o.str();
}

static std::string hex_digest(const unsigned char * p, size_t n) {
    static const char h[] = "0123456789abcdef";
    std::string out(n * 2, '0');
    for (size_t i = 0; i < n; ++i) {
        out[2*i] = h[p[i] >> 4]; out[2*i+1] = h[p[i] & 15];
    }
    return out;
}

static std::string sha256_span(const void * data, size_t size) {
    unsigned char out[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char *>(data), size, out);
    return hex_digest(out, sizeof(out));
}

static std::string sha256_file(const fs::path & path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open for SHA-256 failed: " + path.string());
    SHA256_CTX ctx; SHA256_Init(&ctx);
    std::vector<unsigned char> buffer(8u << 20);
    for (;;) {
        ssize_t n = ::read(fd, buffer.data(), buffer.size());
        if (n < 0) { if (errno == EINTR) continue; ::close(fd); throw std::runtime_error("SHA-256 read failed"); }
        if (n == 0) break;
        SHA256_Update(&ctx, buffer.data(), size_t(n));
    }
    ::close(fd);
    unsigned char out[SHA256_DIGEST_LENGTH]; SHA256_Final(out, &ctx);
    return hex_digest(out, sizeof(out));
}

static void write_all(int fd, const void * data, size_t size) {
    const auto * p = static_cast<const uint8_t *>(data);
    while (size) {
        ssize_t n = ::write(fd, p, size);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("write failed"); }
        p += n; size -= size_t(n);
    }
}

static void pwrite_all(int fd, const void * data, size_t size, uint64_t offset) {
    const auto * p = static_cast<const uint8_t *>(data);
    while (size) {
        ssize_t n = ::pwrite(fd, p, size, off_t(offset));
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("pwrite failed"); }
        p += n; size -= size_t(n); offset += uint64_t(n);
    }
}

static void fsync_dir(const fs::path & dir) {
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw std::runtime_error("open directory failed");
    if (::fsync(fd) != 0) { ::close(fd); throw std::runtime_error("directory fsync failed"); }
    ::close(fd);
}

static void atomic_write(const fs::path & final_path, std::string_view bytes) {
    fs::create_directories(final_path.parent_path());
    fs::path tmp = final_path.string() + ".tmp." + std::to_string(::getpid());
    int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) throw std::runtime_error("create temp failed: " + tmp.string());
    write_all(fd, bytes.data(), bytes.size());
    if (::fsync(fd) != 0) { ::close(fd); throw std::runtime_error("temp fsync failed"); }
    ::close(fd);
    fs::rename(tmp, final_path);
    fsync_dir(final_path.parent_path());
}

struct Mapping {
    int fd = -1;
    size_t size = 0;
    const uint8_t * data = nullptr;
    explicit Mapping(const fs::path & p) {
        fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open source failed");
        struct stat st{}; if (::fstat(fd, &st) != 0) throw std::runtime_error("fstat failed");
        size = size_t(st.st_size);
        data = static_cast<const uint8_t *>(::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0));
        if (data == MAP_FAILED) { data = nullptr; throw std::runtime_error("mmap failed"); }
    }
    ~Mapping() { if (data) ::munmap(const_cast<uint8_t *>(data), size); if (fd >= 0) ::close(fd); }
};

static uint16_t f32_to_f16(float f) {
    uint32_t x = std::bit_cast<uint32_t>(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x7fffffu;
    int exp = int((x >> 23) & 0xffu) - 127 + 15;
    if (((x >> 23) & 0xffu) == 0xffu) {
        if (mant == 0) return uint16_t(sign | 0x7c00u);
        return uint16_t(sign | 0x7e00u | (mant >> 13));
    }
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t q = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1u))) ++q;
        return uint16_t(sign | q);
    }
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    uint32_t q = mant >> 13;
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) {
        if (++q == 0x400u) { q = 0; if (++exp >= 31) return uint16_t(sign | 0x7c00u); }
    }
    return uint16_t(sign | (uint32_t(exp) << 10) | q);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 31u;
    uint32_t mant = h & 1023u;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) x = sign;
        else {
            int e = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
            mant &= 0x3ffu;
            x = sign | (uint32_t(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) x = sign | 0x7f800000u | (mant << 13);
    else x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    return std::bit_cast<float>(x);
}

static int nearest_int(float x) { return int(std::nearbyintf(x)); }

#pragma pack(push, 1)
struct BlockQ2K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; };
#pragma pack(pop)
static_assert(sizeof(BlockQ2K) == 84);

static float make_qkx3(const float * x, const float * weights, uint8_t * L, float & the_min) {
    float min = x[0], max = x[0], sum_w = weights[0], sum_x = weights[0] * x[0];
    for (int i = 1; i < 16; ++i) {
        min = std::min(min, x[i]); max = std::max(max, x[i]);
        float w = weights[i]; sum_w += w; sum_x += w * x[i];
    }
    min = std::min(min, 0.0f);
    if (max <= min) { std::memset(L, 0, 16); the_min = -min; return 0; }
    float iscale = 3.0f / (max - min), scale = 1.0f / iscale, best = 0;
    uint8_t aux[16];
    for (int i = 0; i < 16; ++i) {
        L[i] = uint8_t(std::clamp(nearest_int(iscale * (x[i] - min)), 0, 3));
        float diff = scale * L[i] + min - x[i]; best += weights[i] * diff * diff;
    }
    for (int step = 0; step <= 36; ++step) {
        iscale = (-0.9f + 0.05f * float(step) + 3.0f) / (max - min);
        float sl = 0, sl2 = 0, sxl = 0;
        for (int i = 0; i < 16; ++i) {
            int l = std::clamp(nearest_int(iscale * (x[i] - min)), 0, 3); aux[i] = uint8_t(l);
            float w = weights[i]; sl += w*l; sl2 += w*l*l; sxl += w*l*x[i];
        }
        float D = sum_w*sl2 - sl*sl;
        if (D > 0) {
            float ts = (sum_w*sxl - sum_x*sl)/D;
            float tm = (sl2*sum_x - sl*sxl)/D;
            if (tm > 0) { tm = 0; ts = sxl/sl2; }
            float mad = 0;
            for (int i = 0; i < 16; ++i) { float d = ts*aux[i] + tm - x[i]; mad += weights[i]*d*d; }
            if (mad < best) { std::memcpy(L, aux, 16); best = mad; scale = ts; min = tm; }
        }
    }
    the_min = -min; return scale;
}

static float make_qp(const float * x, const float * weights, uint8_t * L) {
    float max = 0; for (int i=0;i<16;++i) max=std::max(max,x[i]);
    if (max < 1e-15f) { std::memset(L,0,16); return 0; }
    float iscale=15.0f/max, scale=1.0f/iscale, best=0;
    for(int i=0;i<16;++i){L[i]=uint8_t(nearest_int(iscale*x[i])); float d=x[i]-scale*L[i]; best+=weights[i]*d*d;}
    for(int step=-4;step<=4;++step){if(!step)continue; float ti=(0.1f*step+15.0f)/max, ts=1.0f/ti,mse=0;
        for(int i=0;i<16;++i){int l=std::min(15,nearest_int(ti*x[i]));float d=x[i]-ts*l;mse+=weights[i]*d*d;}
        if(mse<best){best=mse;iscale=ti;}}
    float slx=0,sl2=0;
    for(int i=0;i<16;++i){int l=std::min(15,nearest_int(iscale*x[i]));L[i]=uint8_t(l);float w=weights[i];slx+=w*x[i]*l;sl2+=w*l*l;}
    for(int tr=0;tr<5;++tr){int changed=0;for(int i=0;i<16;++i){float w=weights[i],a=slx-w*x[i]*L[i],b=sl2-w*L[i]*L[i];
        if(a>0&&b>0){int nl=std::min(15,nearest_int(x[i]*b/a));if(nl!=L[i]){a+=w*x[i]*nl;b+=w*nl*nl;
            if(a*a*sl2>slx*slx*b){L[i]=uint8_t(nl);slx=a;sl2=b;++changed;}}}}if(!changed)break;}
    return sl2>0?slx/sl2:0;
}

static BlockQ2K quantize_block(const float * x) {
    BlockQ2K y{}; uint8_t L[256], Ls[16], Lm[16]; float scales[16], mins[16], sw[16], weights[16];
    float sumx2=0; for(int i=0;i<256;++i)sumx2+=x[i]*x[i]; float sigma2=sumx2/256.0f;
    for(int j=0;j<16;++j){sw[j]=0;for(int l=0;l<16;++l){float v=x[16*j+l];weights[l]=std::sqrt(sigma2+v*v);sw[j]+=weights[l];}
        scales[j]=make_qkx3(x+16*j,weights,L+16*j,mins[j]);}
    float dm=make_qp(scales,sw,Ls), mm=make_qp(mins,sw,Lm);
    y.d=f32_to_f16(dm);y.dmin=f32_to_f16(mm);dm=f16_to_f32(y.d);mm=f16_to_f32(y.dmin);
    for(int j=0;j<16;++j)y.scales[j]=uint8_t(Ls[j]|(Lm[j]<<4));
    for(int j=0;j<16;++j){float d=dm*(y.scales[j]&15);if(d==0)continue;float m=mm*(y.scales[j]>>4);
        for(int i=0;i<16;++i)L[16*j+i]=uint8_t(std::clamp(nearest_int((x[16*j+i]+m)/d),0,3));}
    for(int j=0;j<256;j+=128)for(int l=0;l<32;++l)y.qs[j/4+l]=uint8_t(L[j+l]|(L[j+l+32]<<2)|(L[j+l+64]<<4)|(L[j+l+96]<<6));
    return y;
}

struct Blas {
    using Gemm = void(*)(int,int,int,int64_t,int64_t,int64_t,double,const double*,int64_t,const double*,int64_t,double,double*,int64_t);
    using SetThreads = void(*)(int);
    void * handle=nullptr; Gemm gemm=nullptr;
    Blas() {
        const char * paths[] = {
          "/home/sean/.local/lib/python3.14/site-packages/numpy.libs/libscipy_openblas64_-017048f4.so",
          nullptr};
        for(int i=0;paths[i]&&!handle;++i)handle=::dlopen(paths[i],RTLD_NOW|RTLD_LOCAL);
        if(handle){gemm=reinterpret_cast<Gemm>(::dlsym(handle,"scipy_cblas_dgemm64_"));
            auto set=reinterpret_cast<SetThreads>(::dlsym(handle,"openblas_set_num_threads_local"));if(set)set(1);}
        if(!gemm)throw std::runtime_error("the NumPy OpenBLAS used by the banked oracle was not found");
    }
    ~Blas(){if(handle)::dlclose(handle);}
    void multiply(const double *a,const double*b,double*c,int64_t m,int64_t n,int64_t k) const {
        gemm(101,111,111,m,n,k,1.0,a,k,b,n,0.0,c,n);
    }
};

static const std::array<double,128*128> & hadamard() {
    static const auto h=[](){std::array<double,128*128>a{};double s=1.0/std::sqrt(128.0);
        for(int i=0;i<128;++i)for(int j=0;j<128;++j)a[i*128+j]=(__builtin_parity(unsigned(i&j))?-s:s);return a;}();
    return h;
}

static const std::array<float,65536> & codebook() {
    static const auto lut=[](){std::array<float,65536>a{};for(uint32_t x=0;x<65536;++x){uint32_t t=x*3417055213u;t=(t&0x8fff8fffu)^0x3b603b60u;
        float v=f16_to_f32(uint16_t(t))+f16_to_f32(uint16_t(t>>16));a[x]=f16_to_f32(f32_to_f16(v));}return a;}();return lut;
}

static float decode_value(const int16_t * packed,int K,int row,int col) {
    static const int dr[8]={9,8,1,0,9,8,1,0};
    static const int dc[8]={8,8,8,8,0,0,0,0};
    const uint32_t * words=reinterpret_cast<const uint32_t*>(packed);
    for(int lane=0;lane<32;++lane){int r0=2*(lane&3),c0=2*((lane>>3)&1)+4*((lane>>4)&1),d=(lane>>2)&1;
        int cur,prev,sh;if(K==2){cur=(lane>>1)&15;prev=(cur-1)&15;sh=(lane&1)?0:16;}
        else{int b=lane*24,r86=b+791,r87=r86&2016;cur=(((r86>>3)&252)-96)/4;prev=lane?((b+755)>>5)-24:23;sh=r87-b-760;}
        uint64_t pair=(uint64_t(words[prev])<<32)|words[cur];
        for(int step=0;step<8;++step)if(r0+dr[step]==row&&c0+dc[step]+d==col){uint16_t w=uint16_t((pair>>(sh+K*step))&0xffffu);return codebook()[w];}}
    throw std::runtime_error("Escha tile mapping hole");
}

struct ShardLayout { std::vector<uint8_t> header; std::array<uint64_t,3> payload_offsets{}; uint64_t size=0; };
static void append_u32(std::vector<uint8_t>&v,uint32_t x){for(int i=0;i<4;++i)v.push_back(uint8_t(x>>(8*i)));}
static void append_u64(std::vector<uint8_t>&v,uint64_t x){for(int i=0;i<8;++i)v.push_back(uint8_t(x>>(8*i)));}
static void append_str(std::vector<uint8_t>&v,std::string_view s){append_u64(v,s.size());v.insert(v.end(),s.begin(),s.end());}

static ShardLayout shard_layout(){ShardLayout l;auto&v=l.header;v.insert(v.end(),{'G','G','U','F'});append_u32(v,3);append_u64(v,3);append_u64(v,1);
    append_str(v,"general.architecture");append_u32(v,8);append_str(v,"qwen35");
    uint64_t rel=0;for(size_t i=0;i<3;++i){auto&s=kSpecs[i];append_str(v,s.output_name);append_u32(v,2);append_u64(v,s.ic);append_u64(v,s.oc);append_u32(v,kQ2K);append_u64(v,rel);rel+=uint64_t(s.oc)*(s.ic/256)*84;rel=(rel+31)&~uint64_t(31);}
    v.resize((v.size()+31)&~size_t(31),0);for(size_t i=0;i<3;++i){l.payload_offsets[i]=v.size();for(size_t j=0;j<i;++j){auto&s=kSpecs[j];l.payload_offsets[i]+=uint64_t(s.oc)*(s.ic/256)*84;l.payload_offsets[i]=(l.payload_offsets[i]+31)&~uint64_t(31);}}
    l.size=v.size()+rel;return l;}

struct PhaseTimes { double reconstruct=0,quantize=0,write=0; };

static void transcode_tensor(const Mapping&src,const TensorSpec&s,int outfd,uint64_t payload_offset,int workers,PhaseTimes&tot) {
    const auto *code=reinterpret_cast<const int16_t*>(src.data+s.code_offset);
    const auto *rin=reinterpret_cast<const uint16_t*>(src.data+s.rin_offset);
    const auto *rout=reinterpret_cast<const uint16_t*>(src.data+s.rout_offset);
    const int itc=s.ic/16,otc=s.oc/16,groups=s.oc/128,rowbytes=(s.ic/256)*84;
    std::atomic<double> rec{0},quant{0},wr{0};std::atomic<bool>failed{false};std::mutex em;std::string error;
    omp_set_dynamic(0);omp_set_num_threads(workers);
#pragma omp parallel for schedule(dynamic,1)
    for(int og=0;og<groups;++og){
      try{
        Blas blas;auto t0=Clock::now();
        std::vector<double>a(size_t(128)*s.ic),first(size_t(128)*s.ic),b(size_t(s.ic)*128),c(size_t(s.ic)*128);
        for(int so=0;so<128;++so){int global_o=og*128+so,ot=global_o/16,oc=global_o%16;
          for(int ic=0;ic<s.ic;++ic){int it=ic/16,ir=ic%16;const int16_t*tile=code+(size_t(it)*otc+ot)*16*s.K;a[size_t(so)*s.ic+ic]=decode_value(tile,s.K,ir,oc);}
          blas.multiply(a.data()+size_t(so)*s.ic,hadamard().data(),first.data()+size_t(so)*s.ic,itc/8,128,128);
        }
        for(int ic=0;ic<s.ic;++ic){double r=f16_to_f32(rin[ic]);for(int so=0;so<128;++so)b[size_t(ic)*128+so]=first[size_t(so)*s.ic+ic]*r;}
        blas.multiply(b.data(),hadamard().data(),c.data(),s.ic,128,128);
        auto t1=Clock::now();
        std::vector<float>row(s.ic);std::vector<BlockQ2K>packed(size_t(128)*(s.ic/256));
        for(int to=0;to<128;++to){float ro=f16_to_f32(rout[og*128+to]);for(int ic=0;ic<s.ic;++ic)row[ic]=float(c[size_t(ic)*128+to]*ro);
          for(int ib=0;ib<s.ic/256;++ib)packed[size_t(to)*(s.ic/256)+ib]=quantize_block(row.data()+ib*256);}
        auto t2=Clock::now();pwrite_all(outfd,packed.data(),packed.size()*sizeof(BlockQ2K),payload_offset+uint64_t(og)*128*rowbytes);auto t3=Clock::now();
        rec.fetch_add(seconds(t0,t1));quant.fetch_add(seconds(t1,t2));wr.fetch_add(seconds(t2,t3));
      }catch(const std::exception&e){failed=true;std::lock_guard<std::mutex>g(em);error=e.what();}
    }
    if(failed)throw std::runtime_error(error);tot.reconstruct+=rec.load()/workers;tot.quantize+=quant.load()/workers;tot.write+=wr.load()/workers;
}

static std::array<std::string,3> payload_hashes(const Mapping&m,const ShardLayout&l){std::array<std::string,3>h{};for(size_t i=0;i<3;++i){auto&s=kSpecs[i];size_t n=size_t(s.oc)*(s.ic/256)*84;h[i]=sha256_span(m.data+l.payload_offsets[i],n);}return h;}

static std::string receipt_json(const std::string&shard_sha,const std::array<std::string,3>&p,const ShardLayout&l){std::ostringstream o;o<<"{\"layer\":0,\"native_abi\":\""<<kNativeAbi<<"\",\"oracle_abi\":\""<<kOracleAbi<<"\",\"quantizer_abi\":\""<<kQuantAbi<<"\",\"recipe\":\"all-q2_k-layer-shard-v1\",\"shard_sha256\":\""<<shard_sha<<"\",\"source_sha256\":\""<<kSourceSha<<"\",\"tensors\":[";
    for(size_t i=0;i<3;++i){if(i)o<<',';auto&s=kSpecs[i];o<<"{\"byte_count\":"<<uint64_t(s.oc)*(s.ic/256)*84<<",\"data_offset\":"<<l.payload_offsets[i]<<",\"name\":\""<<s.output_name<<"\",\"payload_sha256\":\""<<p[i]<<"\",\"shape\":["<<s.oc<<','<<s.ic<<"]}";}
    o<<"]}\n";return o.str();}

static bool valid_published(const fs::path&shard,const fs::path&receipt,const ShardLayout&l){
    if(!fs::is_regular_file(shard)||!fs::is_regular_file(receipt)||fs::file_size(shard)!=l.size)return false;
    try{Mapping m(shard);if(std::memcmp(m.data,l.header.data(),l.header.size())!=0)return false;auto p=payload_hashes(m,l);for(size_t i=0;i<3;++i)if(p[i]!=kSpecs[i].expected_payload_sha)return false;
        std::string ss=sha256_file(shard),expect=receipt_json(ss,p,l);std::ifstream f(receipt,std::ios::binary);std::string got((std::istreambuf_iterator<char>(f)),{});return got==expect;}catch(...){return false;}
}

static long rss_kb(){struct rusage r{};getrusage(RUSAGE_SELF,&r);return r.ru_maxrss;}
static long current_rss_kb(){std::ifstream f("/proc/self/status");std::string k;while(f>>k){if(k=="VmRSS:"){long n;std::string unit;f>>n>>unit;return n;}std::string rest;std::getline(f,rest);}return 0;}

static int command_quantize_raw(int argc,char**argv){if(argc!=4)throw std::runtime_error("quantize-raw INPUT OUTPUT");Mapping m(argv[2]);if(m.size%1024)throw std::runtime_error("raw fp32 size must be a multiple of 256 values");size_t nb=m.size/1024;std::vector<BlockQ2K>out(nb);auto*x=reinterpret_cast<const float*>(m.data);for(size_t i=0;i<nb;++i)out[i]=quantize_block(x+256*i);int fd=::open(argv[3],O_CREAT|O_TRUNC|O_WRONLY,0644);write_all(fd,out.data(),out.size()*84);::fsync(fd);::close(fd);return 0;}

static int command_prepare(int argc,char**argv){fs::path source,cache,report;int workers=0;for(int i=2;i<argc;++i){std::string a=argv[i];if(a=="--source"&&i+1<argc)source=argv[++i];else if(a=="--cache-dir"&&i+1<argc)cache=argv[++i];else if(a=="--report"&&i+1<argc)report=argv[++i];else if(a=="--workers"&&i+1<argc)workers=std::stoi(argv[++i]);else throw std::runtime_error("unknown/missing prepare argument: "+a);}
    if(fs::weakly_canonical(source)!=fs::weakly_canonical(kSourcePath))throw std::runtime_error("Slice 1 accepts only the canonical source GGUF");if(workers<1||workers>16)throw std::runtime_error("workers must be 1..16");
    auto wall0=Clock::now(),hash0=Clock::now();std::string source_sha=sha256_file(source);auto hash1=Clock::now();if(source_sha!=kSourceSha)throw std::runtime_error("canonical source SHA-256 mismatch");
    fs::path layers=cache/"layers";fs::create_directories(layers);fs::path shard=layers/"blk.000.gguf",receipt=layers/"blk.000.receipt.json";ShardLayout layout=shard_layout();long idle=current_rss_kb();std::string action;PhaseTimes phases{};double verify_seconds=0;
    if(valid_published(shard,receipt,layout))action="skipped_valid";else{action=(fs::exists(shard)||fs::exists(receipt))?"rebuilt_invalid":"built";fs::path tmp=shard.string()+".tmp."+std::to_string(::getpid());int fd=::open(tmp.c_str(),O_CREAT|O_TRUNC|O_RDWR,0644);if(fd<0)throw std::runtime_error("create shard temp failed");if(::ftruncate(fd,off_t(layout.size))!=0)throw std::runtime_error("ftruncate failed");pwrite_all(fd,layout.header.data(),layout.header.size(),0);
        Mapping src(source);auto trans0=Clock::now();for(size_t i=0;i<3;++i)transcode_tensor(src,kSpecs[i],fd,layout.payload_offsets[i],workers,phases);auto trans1=Clock::now();(void)trans0;(void)trans1;if(::fsync(fd)!=0)throw std::runtime_error("shard fsync failed");::close(fd);
        auto v0=Clock::now();Mapping built(tmp);if(std::memcmp(built.data,layout.header.data(),layout.header.size())!=0)throw std::runtime_error("shard header verification failed");auto ph=payload_hashes(built,layout);for(size_t i=0;i<3;++i)if(ph[i]!=kSpecs[i].expected_payload_sha)throw std::runtime_error(std::string("STOP: native payload differs from oracle for ")+kSpecs[i].role+": "+ph[i]);std::string shard_sha=sha256_file(tmp);auto v1=Clock::now();verify_seconds=seconds(v0,v1);fs::rename(tmp,shard);fsync_dir(layers);atomic_write(receipt,receipt_json(shard_sha,ph,layout));if(!valid_published(shard,receipt,layout))throw std::runtime_error("published layer failed independent resume validation");}
    auto wall1=Clock::now();long peak=rss_kb(),now=current_rss_kb();double hash_s=seconds(hash0,hash1),wall_s=seconds(wall0,wall1);double scalable=wall_s-hash_s;double values=3.0*89128960.0;double vps=action=="skipped_valid"?0:values/std::max(1e-9,scalable);double projection=action=="skipped_valid"?0:hash_s+17112760320.0/vps;uint64_t worker_buf=uint64_t(kSpecs[2].ic)*128*(sizeof(double)*4+sizeof(float))+uint64_t(128)*(kSpecs[2].ic/256)*84;
    std::ostringstream o;o<<std::fixed<<std::setprecision(6)<<"{\n  \"action\": \""<<action<<"\",\n  \"absolute_peak_rss_kb\": "<<peak<<",\n  \"incremental_peak_rss_kb\": "<<std::max<long>(0,peak-idle)<<",\n  \"idle_rss_kb\": "<<idle<<",\n  \"per_worker_buffer_bytes\": "<<worker_buf<<",\n  \"source_hash_seconds\": "<<hash_s<<",\n  \"reconstruction_seconds\": "<<phases.reconstruct<<",\n  \"quantization_seconds\": "<<phases.quantize<<",\n  \"disk_write_seconds\": "<<phases.write<<",\n  \"verify_hash_seconds\": "<<verify_seconds<<",\n  \"wall_seconds\": "<<wall_s<<",\n  \"aggregate_values_per_second\": "<<vps<<",\n  \"projected_all64_seconds\": "<<projection<<",\n  \"wall_budget_holds\": "<<(projection<=120?"true":"false")<<",\n  \"rss_budget_holds\": "<<((peak-idle)<=1024L*1024L?"true":"false")<<",\n  \"workers\": "<<workers<<",\n  \"current_rss_kb\": "<<now<<"\n}\n";atomic_write(report,o.str());std::cout<<o.str();return 0;}

int main(int argc,char**argv){try{std::fesetround(FE_TONEAREST);if(argc<2)throw std::runtime_error("usage: escha-native-transcode quantize-raw|prepare ...");std::string c=argv[1];if(c=="quantize-raw")return command_quantize_raw(argc,argv);if(c=="prepare")return command_prepare(argc,argv);throw std::runtime_error("unknown command");}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
