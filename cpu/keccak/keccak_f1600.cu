#include <string.h>
#include "keccak_f1600.cuh"

#ifdef __CUDACC__
__device__ __constant__ uint64_t d_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};
#endif

static const uint64_t h_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

#if defined(__CUDA_ARCH__)
  #define RC d_RC
#else
  #define RC h_RC
#endif

#define ROTL64(x, y) (((y) == 0) ? (x) : (((x) << (y)) | ((x) >> (64 - (y)))))

__host__ __device__
void keccakf1600_permute(uint64_t state[25])
{
    uint64_t C[5], D[5];

    #pragma unroll 1
    for (int r = 0; r < KECCAK1600_NR_ROUNDS; r++) {

        #pragma unroll
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        #pragma unroll
        for (int i = 0; i < 5; i++)
            D[i] = C[(i+4)%5] ^ ROTL64(C[(i+1)%5], 1);
        #pragma unroll
        for (int i = 0; i < 25; i++)
            state[i] ^= D[i%5];

        uint64_t tmp[25];
        tmp[ 0] = ROTL64(state[ 0],  0);
        tmp[10] = ROTL64(state[ 1],  1);
        tmp[20] = ROTL64(state[ 2], 62);
        tmp[ 5] = ROTL64(state[ 3], 28);
        tmp[15] = ROTL64(state[ 4], 27);
        tmp[16] = ROTL64(state[ 5], 36);
        tmp[ 1] = ROTL64(state[ 6], 44);
        tmp[11] = ROTL64(state[ 7],  6);
        tmp[21] = ROTL64(state[ 8], 55);
        tmp[ 6] = ROTL64(state[ 9], 20);
        tmp[ 7] = ROTL64(state[10],  3);
        tmp[17] = ROTL64(state[11], 10);
        tmp[ 2] = ROTL64(state[12], 43);
        tmp[12] = ROTL64(state[13], 25);
        tmp[22] = ROTL64(state[14], 39);
        tmp[23] = ROTL64(state[15], 41);
        tmp[ 8] = ROTL64(state[16], 45);
        tmp[18] = ROTL64(state[17], 15);
        tmp[ 3] = ROTL64(state[18], 21);
        tmp[13] = ROTL64(state[19],  8);
        tmp[14] = ROTL64(state[20], 18);
        tmp[24] = ROTL64(state[21],  2);
        tmp[ 9] = ROTL64(state[22], 61);
        tmp[19] = ROTL64(state[23], 56);
        tmp[ 4] = ROTL64(state[24], 14);

        #pragma unroll
        for (int j = 0; j < 25; j += 5)
            #pragma unroll
            for (int i = 0; i < 5; i++)
                state[j+i] = tmp[j+i] ^ ((~tmp[j+(i+1)%5]) & tmp[j+(i+2)%5]);

        state[0] ^= RC[r];
    }
}

__host__ __device__
void keccak1600_hash(const uint8_t *msg, size_t msg_len,
                     uint8_t digest[KECCAK1600_HASH_BYTES])
{
    uint64_t state[25];
    uint8_t *st = (uint8_t *)state;

    #pragma unroll
    for (int i = 0; i < 25; i++) state[i] = 0;

    for (size_t i = 0; i < msg_len; i++) st[i] ^= msg[i];

    st[msg_len]                   ^= 0x06;
    st[KECCAK1600_RATE_BYTES - 1] ^= 0x80;

    keccakf1600_permute(state);

    for (int i = 0; i < KECCAK1600_HASH_BYTES; i++) digest[i] = st[i];
}

#ifdef __CUDACC__
void keccak_f1600_init_cuda(void) {}
#endif

#undef RC
