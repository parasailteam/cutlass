//rungemm.cu
#include<overlaphandle.h>

int main() {
    OverlapHandle handle(N, M, K, TilesX, TilesY, TilesZ)

    handle.setThreadBlockOrder();

    while(iter <= epochs)
        handle.setproducer();
        Gemm<producer>(N, M, K, TileX, TileY, handle);
        handle.setconsumer();
        Gemm<consumer>(N, M, K, TileX, TileY, handle);
}

template<typename isProducerOrConsumer>
GemmKernel(handle) {
    tilex, tiley, tilez = handle.getTileIds();

    if (isConsumer) {
        handle.waitOnRow(tilex, tiley, tilez);
    }

    //process tile
    if (isProducer) {
        handle.setRowStatus(tilex, tiley, tilez)
    }
}

GemmKernel(handle) {
    tilex, tiley, tilez = handle.getTileIds();

    if (isConsumer) {
        for (int k = 0; k < K; k += TileK) {
            handle.waitOnTile(tilex, tiley, tilez);
            loadsA to shared
            load Bto shared
            perform matmul
        }

        write tile to global memory for C
    }

    //process tile
    if (isProducer) {
        handle.setTileStatus(tilex, tiley, tilez)
    }
}