#define SCCL_MAX_NUM_STEPS 512
#define SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL 32

#define SCCL_INPUT_BUFFER 0
#define SCCL_OUTPUT_BUFFER 1
#define SCCL_SCRATCH_BUFFER 2

#define SCCL_SEND 0
#define SCCL_RECV 1
#define SCCL_RECV_COPY_SEND 2
#define SCCL_RECV_REDUCE_SEND 3
#define SCCL_RECV_REDUCE_COPY 4
#define SCCL_RECV_REDUCE_COPY_SEND 5
#define SCCL_NO_OP 6

// TODO: compress this by a lot!
struct scclTransfer {
  int16_t srcoffset;
  int16_t dstoffset;
  uint8_t srcbuffer; // follow SCCL_THIS_INPUT/SCCL_THIS_OUTPUT macros
  uint8_t dstbuffer; // follow SCCL_THIS_INPUT/SCCL_THIS_OUTPUT macros
  int8_t dependentBid; // -1 if not dependent on any threadblock
  int16_t dependentStep;
  int8_t has_dependence;
  uint8_t type;
  uint8_t count;
};

struct scclThreadBlock {
  int8_t sendpeer;
  int8_t recvpeer;
  uint16_t nsteps;
  uint8_t channelId; // associated channel
  uint16_t rid; // relative id of this thread block to the channel
  // step is used to index into this array. transfers[step] is the addr to transfer.
  struct scclTransfer transfers[SCCL_MAX_NUM_STEPS];
};

#define SCCL_MAX_COUNT 16

struct scclChannelInfo {
  int sendPeers[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // nchunksForSendPeer[i][j] represents the number of times chunks are sent in counts of j-1 for threadblock i. we do not keep counts of 0.
  int nchunksForSendPeer[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][SCCL_MAX_COUNT];
  int nsendPeers;
  int recvPeers[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  int nchunksForRecvPeer[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][SCCL_MAX_COUNT];
  int nrecvPeers;
  int nBlocksForChannel;
};

struct scclFlag {
  uint64_t flag;
  uint64_t align[3]; // To avoid false sharing
};

// gpuId is the one that is in comm->rank
struct scclAlgorithm {
  // max(#chunks in input, #chunks in output)
  int nchunksPerLoop;
  // the protocol that the algorithm needs to use
  int protocol;
  // total number of threadblocks needed by SCCL algorithm
  int nBlocks; // TODO could be removed
  // bid is used as an index into this array
  struct scclThreadBlock scclTB[MAXCHANNELS*SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // number of channels needed by SCCL algorithm
  int nChannels;
  // the arrays in this struct can be inferred from scclTB. they are created to use NCCL API easily
  struct scclChannelInfo scclChannels[MAXCHANNELS];
  // number of scratch chunks that SCCL will use
  int nScratchChunks;
  // declaration for scratchBuffer. This is only to be accessed by the host
  size_t scratchBufferSize;
  void* scratchBuffer;
  //Reduction Operator. If the algorithm performs reduction it will specify the reduction operator.
  //If the algorithm do not perform reduction, its reduction operator is considered as ncclSum.
  int redOp; //TODO: Yes, enum is represented as an int usually.

  // allocate enough SCCL flags (SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL * MAXCHANNELS) to synchronize across thread blocks
  struct scclFlag* flags;
  // this flag is used to indicate we have we have looped around the channels work queue. Once that happens, the flags need to be reset.
  int flagsNeedReset;
  //Size of the leading dimension of the chunk
  int chunkld;
};