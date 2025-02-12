#include <algorithm>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <numeric>

#include "realm/logging.h"
#include "bootstrap/bootstrap.h"
#include "ucc_comm.h"

namespace Realm {
  Logger log_ucc("ucc");

  namespace ucc {
    UCCComm::UCCComm(int _rank, int _world_sz, bootstrap_handle_t *bh)
      : rank(_rank)
      , world_sz(_world_sz)
    {
      oob_comm = std::make_unique<OOBGroupComm>(rank, world_sz, bh);
    };

    ucc_status_t UCCComm::init_lib()
    {
      ucc_lib_config_h lib_config;
      ucc_lib_params_t lib_params;
      ucc_status_t status;

      if(status = ucc_lib_config_read(/* env_prefix */ nullptr,
                                      /* env_prefix */ nullptr, &lib_config),
         UCC_OK != status) {
        log_ucc.error() << "Failed to read the library configuration\n";
        return status;
      }

      std::memset(&lib_params, 0, sizeof(ucc_lib_params_t));
      lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
      lib_params.thread_mode = UCC_THREAD_MULTIPLE;
      lib_params.coll_types = {};
      lib_params.reduction_types = {};
      lib_params.sync_type = {};

      if(status = ucc_init(&lib_params, lib_config, &lib), UCC_OK != status) {
        log_ucc.error() << "UCCLayer : Failed to initialize the ucc library\n";
        ucc_lib_config_release(lib_config);
        return status;
      }

      log_ucc.info() << "UCC library configured successfully\n";
      ucc_lib_config_release(lib_config);
      return UCC_OK;
    }

    ucc_status_t UCCComm::create_context()
    {
      ucc_context_config_h ctx_config;
      ucc_context_params_t ctx_params;
      ucc_status_t status;

      if(status = ucc_context_config_read(lib, NULL, &ctx_config), UCC_OK != status) {
        log_ucc.error() << "Failed to read context config\n";
        ucc_finalize(lib);
        return status;
      }
      std::memset(&ctx_params, 0, sizeof(ucc_context_params_t));
      ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
      ctx_params.type = UCC_CONTEXT_SHARED;
      ctx_params.oob.allgather = ucc::OOBGroupComm::oob_allgather;
      ctx_params.oob.req_test = ucc::OOBGroupComm::oob_allgather_test;
      ctx_params.oob.req_free = ucc::OOBGroupComm::oob_allgather_free;

      ctx_params.oob.coll_info = oob_comm->get_coll_info();

      ctx_params.oob.n_oob_eps = static_cast<uint32_t>(oob_comm->get_world_size());
      ctx_params.oob.oob_ep = static_cast<uint32_t>(oob_comm->get_rank());

      if(status = ucc_context_create(lib, &ctx_params, ctx_config, &context),
         UCC_OK != status) {
        log_ucc.error() << "UCCComm : Failed to create ucc context\n";
        ucc_context_config_release(ctx_config);
        ucc_finalize(lib);
        return status;
      }

      log_ucc.info() << "UCC Context created successfully\n";

      ucc_context_config_release(ctx_config);
      return UCC_OK;
    }

    ucc_status_t UCCComm::create_team()
    {
      ucc_team_params team_params;
      ucc_status_t status;

      team_params.mask = UCC_TEAM_PARAM_FIELD_OOB;
      team_params.ordering = UCC_COLLECTIVE_POST_ORDERED;
      team_params.oob.coll_info = oob_comm->get_coll_info();

      team_params.oob.allgather = ucc::OOBGroupComm::oob_allgather;
      team_params.oob.req_test = ucc::OOBGroupComm::oob_allgather_test;
      team_params.oob.req_free = ucc::OOBGroupComm::oob_allgather_free;

      team_params.oob.n_oob_eps = static_cast<uint32_t>(world_sz);
      team_params.oob.oob_ep = static_cast<uint32_t>(rank);

      if(status = ucc_team_create_post(&context, 1, &team_params, &team),
         UCC_OK != status) {
        log_ucc.error() << "Failed to post team creation request\n";
        ucc_context_destroy(context);
        ucc_finalize(lib);
        return status;
      }

      while((status = ucc_team_create_test(team)) == UCC_INPROGRESS) {
        ucc_context_progress(context);
      }

      if(status == UCC_OK) {
        log_ucc.info() << "UCC Team created successfully.\n"
                       << "My rank is " << get_rank() << ", world size is "
                       << get_world_size() << "\n";
      } else {
        log_ucc.error() << "UCC Team creation failed.\n";
        UCC_Finalize();
      }

      return status;
    }

    ucc_status_t UCCComm::init()
    {
      ucc_status_t status{UCC_OK};

      status = init_lib();
      if(UCC_OK != status) {
        return status;
      }

      status = create_context();
      if(UCC_OK != status) {
        return status;
      }

      status = create_team();
      if(UCC_OK != status) {
        return status;
      }
      return status;
    }

    inline void UCCComm::ucc_check(const ucc_status_t &status)
    {
      if(status < 0) {
        std::cerr << "UCC: Failed " << std::string(ucc_status_string(status))
                  << std::endl;
        assert(0);
      }
    }

    ucc_status_t UCCComm::ucc_collective(ucc_coll_args_t &coll_args, ucc_coll_req_h &req)
    {
      ucc_status_t status;
      status = ucc_collective_init(&coll_args, &req, team);
      ucc_check(status);

      status = ucc_collective_post(req);
      ucc_check(status);

      while(UCC_OK != (status = ucc_collective_test(req))) {
        ucc_check(status);
        status = ucc_context_progress(context);
        ucc_check(status);
      }

      status = ucc_collective_finalize(req);
      ucc_check(status);
      return status;
    }

    ucc_status_t UCCComm::UCC_Bcast(void *buffer, int count, ucc_datatype_t datatype,
                                    int root)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_BCAST;
      coll_args.root = root;
      coll_args.src.info.buffer = buffer;
      coll_args.src.info.count = count;
      coll_args.src.info.datatype = datatype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

      return ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Gather(void *sbuf, int sendcount, ucc_datatype_t sendtype,
                                     void *rbuf, int recvcount, ucc_datatype_t recvtype,
                                     int root)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_GATHER;
      coll_args.root = root;
      coll_args.src.info.buffer = sbuf;
      coll_args.src.info.count = sendcount;
      coll_args.src.info.datatype = sendtype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

      if(rank == root) {
        coll_args.dst.info.buffer = rbuf;
        coll_args.dst.info.count = recvcount;
        coll_args.dst.info.datatype = recvtype;
        coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
      }

      return ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Allgather(void *sbuf, int sendcount,
                                        ucc_datatype_t sendtype, void *rbuf,
                                        int recvcount, ucc_datatype_t recvtype)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_ALLGATHER;
      coll_args.src.info.buffer = sbuf;
      coll_args.src.info.count = sendcount;
      coll_args.src.info.datatype = sendtype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

      coll_args.dst.info.buffer = rbuf;
      coll_args.dst.info.count = recvcount;
      coll_args.dst.info.datatype = recvtype;
      coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

      return ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Allreduce(void *sbuf, void *rbuf, int count,
                                        ucc_datatype_t datatype, ucc_reduction_op_t op)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
      coll_args.src.info.buffer = const_cast<void *>(sbuf);
      coll_args.src.info.count = count;
      coll_args.src.info.datatype = datatype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
      coll_args.dst.info.buffer = rbuf;
      coll_args.dst.info.count = count;
      coll_args.dst.info.datatype = datatype;
      coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
      coll_args.op = op;

      return ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Allgatherv(void *sbuf, int count, ucc_datatype_t sendtype,
                                         void *rbuf, const std::vector<int> &recvcounts,
                                         const std::vector<int> &displs,
                                         ucc_datatype_t recvtype)
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
      coll_args.src.info.buffer = const_cast<void *>(sbuf);
      coll_args.src.info.count = recvcounts[rank];
      coll_args.src.info.datatype = sendtype;
      coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
      coll_args.dst.info_v.buffer = rbuf;
      coll_args.dst.info_v.counts = (ucc_count_t *)(recvcounts.data());
      coll_args.dst.info_v.displacements = (ucc_aint_t *)(displs.data());
      coll_args.dst.info_v.datatype = recvtype;
      coll_args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

      return ucc_collective(coll_args, req);
    }

    ucc_status_t UCCComm::UCC_Barrier()
    {
      ucc_coll_req_h req;
      ucc_coll_args_t coll_args;

      coll_args.mask = 0;
      coll_args.coll_type = UCC_COLL_TYPE_BARRIER;

      ucc_status_t status;
      status = ucc_collective_init(&coll_args, &req, team);
      ucc_check(status);

      status = ucc_collective_post(req);
      ucc_check(status);

      status = ucc_collective_test(req);
      while(status > UCC_OK) {
        status = ucc_context_progress(context);
        assert(status >= UCC_OK); // UCC_OK >= 0
        status = ucc_collective_test(req);
      }
      ucc_check(status);

      status = ucc_collective_finalize(req);
      ucc_check(status);
      return status;
    }

    ucc_status_t UCCComm::UCC_Finalize()
    {
      ucc_status_t st{UCC_OK};
      do {
        st = ucc_team_destroy(team);
      } while(st == UCC_INPROGRESS);
      if(st != UCC_OK) {
        std::cerr << "ucc team destroy error: " << std::string(ucc_status_string(st));
      }
      ucc_context_destroy(context);
      ucc_finalize(lib);

      return UCC_OK;
    }
  } // namespace ucc
} // namespace Realm
