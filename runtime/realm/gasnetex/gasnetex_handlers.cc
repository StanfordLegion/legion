/* Copyright 2022 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// GASNet-EX network module message handlers

#include "realm/gasnetex/gasnetex_handlers.h"

namespace Realm {

#ifdef DEBUG_REALM
  // add a way to add artifical backpressure to immediate sends
  atomic<int> dbg_artifical_backpressure_pct(-1);

  static bool check_artificial_backpressure()
  {
    int chance = dbg_artifical_backpressure_pct.load();
    if(chance < 0) {
      const char *e = getenv("REALM_GEX_ARTIFICIAL_BACKPRESSURE");
      if(e) {
	chance = atoi(e);
	dbg_artifical_backpressure_pct.store(chance);
      }
    }
    if(chance > 0) {
      int r = rand() % 100;
      return (r < chance);
    } else
      return false;
  }
#endif

  enum {
    HIDX_COMPREPLY_BASE = GEX_AM_INDEX_BASE,
    HIDX_COMPREPLY_MAX = HIDX_COMPREPLY_BASE + 16,
    HIDX_LONG_AS_GET,
    HIDX_PUT_HEADER,
    HIDX_SHORTREQ_BASE,
    HIDX_SHORTREQ_MAX = HIDX_SHORTREQ_BASE + 16,
    HIDX_MEDREQ_BASE,
    HIDX_MEDREQ_MAX = HIDX_MEDREQ_BASE + 16,
    HIDX_LONGREQ_BASE,
    HIDX_LONGREQ_MAX = HIDX_LONGREQ_BASE + 16,
    HIDX_BATCHREQ,
  };

#define HIDX_COMPREPLY(narg) (HIDX_COMPREPLY_BASE + (narg))
#define HIDX_SHORTREQ(narg) (HIDX_SHORTREQ_BASE + (narg))
#define HIDX_MEDREQ(narg) (HIDX_MEDREQ_BASE + (narg))
#define HIDX_LONGREQ(narg) (HIDX_LONGREQ_BASE + (narg))

  template <typename T>
  static gex_AM_Arg_t ARG_LO(T val)
  {
    return (reinterpret_cast<uintptr_t>(val) & 0xFFFFFFFFU);
  }

  template <typename T>
  static gex_AM_Arg_t ARG_HI(T val)
  {
    return (reinterpret_cast<uintptr_t>(val) >> 32);
  }

  static uintptr_t ARG_COMBINE(gex_AM_Arg_t lo, gex_AM_Arg_t hi)
  {
    return ((uintptr_t(lo) & 0xFFFFFFFFU) |
	    (uintptr_t(hi) << 32));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // completion reply
  //

  namespace GASNetEXHandlers {

    int send_completion_reply(gex_EP_t src_ep,
			      gex_Rank_t tgt_rank,
			      gex_EP_Index_t tgt_ep_index,
			      const gex_AM_Arg_t *args,
			      size_t nargs,
			      gex_Flags_t flags)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return 1;
#endif

      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);

      switch(nargs) {
      case 1: return gex_AM_RequestShort1(pair, tgt_rank, HIDX_COMPREPLY(1), flags,
					  args[0]);
      case 2: return gex_AM_RequestShort2(pair, tgt_rank, HIDX_COMPREPLY(2), flags,
					  args[0], args[1]);
      case 3: return gex_AM_RequestShort3(pair, tgt_rank, HIDX_COMPREPLY(3), flags,
					  args[0], args[1], args[2]);
      case 4: return gex_AM_RequestShort4(pair, tgt_rank, HIDX_COMPREPLY(4), flags,
					  args[0], args[1], args[2], args[3]);
      case 5: return gex_AM_RequestShort5(pair, tgt_rank, HIDX_COMPREPLY(5), flags,
					  args[0], args[1], args[2], args[3],
					  args[4]);
      case 6: return gex_AM_RequestShort6(pair, tgt_rank, HIDX_COMPREPLY(6), flags,
					  args[0], args[1], args[2], args[3],
					  args[4], args[5]);
      case 7: return gex_AM_RequestShort7(pair, tgt_rank, HIDX_COMPREPLY(7), flags,
					  args[0], args[1], args[2], args[3],
					  args[4], args[5], args[6]);
      case 8: return gex_AM_RequestShort8(pair, tgt_rank, HIDX_COMPREPLY(8), flags,
					  args[0], args[1], args[2], args[3],
					  args[4], args[5], args[6], args[7]);
      case 9: return gex_AM_RequestShort9(pair, tgt_rank, HIDX_COMPREPLY(9), flags,
					  args[0], args[1], args[2], args[3],
					  args[4], args[5], args[6], args[7],
					  args[8]);
      case 10: return gex_AM_RequestShort10(pair, tgt_rank, HIDX_COMPREPLY(10), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8], args[9]);
      case 11: return gex_AM_RequestShort11(pair, tgt_rank, HIDX_COMPREPLY(11), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8], args[9], args[10]);
      case 12: return gex_AM_RequestShort12(pair, tgt_rank, HIDX_COMPREPLY(12), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8], args[9], args[10], args[11]);
      case 13: return gex_AM_RequestShort13(pair, tgt_rank, HIDX_COMPREPLY(13), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8], args[9], args[10], args[11],
					    args[12]);
      case 14: return gex_AM_RequestShort14(pair, tgt_rank, HIDX_COMPREPLY(14), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8], args[9], args[10], args[11],
					    args[12], args[13]);
      case 15: return gex_AM_RequestShort15(pair, tgt_rank, HIDX_COMPREPLY(15), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8], args[9], args[10], args[11],
					    args[12], args[13], args[14]);
      case 16: return gex_AM_RequestShort16(pair, tgt_rank, HIDX_COMPREPLY(16), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8], args[9], args[10], args[11],
					    args[12], args[13], args[14], args[15]);
      default:
	{
	  fprintf(stderr, "ERROR: send_completion_reply with nargs=%zd\n", nargs);
	  abort();
	}
      }
      return 0;
    }

  };

  static void handle_completion_reply(gex_Token_t token,
				      const gex_AM_Arg_t *args,
				      size_t nargs)
  {
    gex_Token_Info_t info;
    // ask for srcrank and ep - both are required, so no need to check result
    gex_Token_Info(token, &info, (GEX_TI_SRCRANK | GEX_TI_EP));

    // ask the ep for the 'internal' pointer
    void *cdata = gex_EP_QueryCData(info.gex_ep);
    GASNetEXInternal *internal = static_cast<GASNetEXInternal *>(cdata);

    internal->handle_completion_reply(info.gex_srcrank, args, nargs);
  }

  static void handle_completion_reply_1(gex_Token_t token,
					gex_AM_Arg_t arg0)
  {
    gex_AM_Arg_t args[1];
    args[0] = arg0;
    handle_completion_reply(token, args, 1);
  }

  static void handle_completion_reply_2(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1)
  {
    gex_AM_Arg_t args[2];
    args[0] = arg0; args[1] = arg1;
    handle_completion_reply(token, args, 2);
  }

  static void handle_completion_reply_3(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2)
  {
    gex_AM_Arg_t args[3];
    args[0] = arg0; args[1] = arg1; args[2] = arg2;
    handle_completion_reply(token, args, 3);
  }

  static void handle_completion_reply_4(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3)
  {
    gex_AM_Arg_t args[4];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    handle_completion_reply(token, args, 4);
  }

  static void handle_completion_reply_5(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					gex_AM_Arg_t arg4)
 {
    gex_AM_Arg_t args[5];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4;
    handle_completion_reply(token, args, 5);
  }

  static void handle_completion_reply_6(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					gex_AM_Arg_t arg4, gex_AM_Arg_t arg5)
 {
    gex_AM_Arg_t args[6];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5;
    handle_completion_reply(token, args, 6);
  }

  static void handle_completion_reply_7(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6)
 {
    gex_AM_Arg_t args[7];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6;
    handle_completion_reply(token, args, 7);
  }

  static void handle_completion_reply_8(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7)
 {
    gex_AM_Arg_t args[8];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    handle_completion_reply(token, args, 8);
  }

  static void handle_completion_reply_9(gex_Token_t token,
					gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					gex_AM_Arg_t arg8)
 {
    gex_AM_Arg_t args[9];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8;
    handle_completion_reply(token, args, 9);
  }

  static void handle_completion_reply_10(gex_Token_t token,
					 gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					 gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					 gex_AM_Arg_t arg8, gex_AM_Arg_t arg9)
 {
    gex_AM_Arg_t args[10];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8; args[9] = arg9;
    handle_completion_reply(token, args, 10);
  }

  static void handle_completion_reply_11(gex_Token_t token,
					 gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					 gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					 gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10)
 {
    gex_AM_Arg_t args[11];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8; args[9] = arg9; args[10] = arg10;
    handle_completion_reply(token, args, 11);
  }

  static void handle_completion_reply_12(gex_Token_t token,
					 gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					 gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					 gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11)
 {
    gex_AM_Arg_t args[12];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8; args[9] = arg9; args[10] = arg10; args[11] = arg11;
    handle_completion_reply(token, args, 12);
  }

  static void handle_completion_reply_13(gex_Token_t token,
					 gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					 gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					 gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
					 gex_AM_Arg_t arg12)
 {
    gex_AM_Arg_t args[13];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8; args[9] = arg9; args[10] = arg10; args[11] = arg11;
    args[12] = arg12;
    handle_completion_reply(token, args, 13);
  }

  static void handle_completion_reply_14(gex_Token_t token,
					 gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					 gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					 gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
					 gex_AM_Arg_t arg12, gex_AM_Arg_t arg13)
 {
    gex_AM_Arg_t args[14];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8; args[9] = arg9; args[10] = arg10; args[11] = arg11;
    args[12] = arg12; args[13] = arg13;
    handle_completion_reply(token, args, 14);
  }

  static void handle_completion_reply_15(gex_Token_t token,
					 gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					 gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					 gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
					 gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14)
 {
    gex_AM_Arg_t args[15];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8; args[9] = arg9; args[10] = arg10; args[11] = arg11;
    args[12] = arg12; args[13] = arg13; args[14] = arg14;
    handle_completion_reply(token, args, 15);
  }

  static void handle_completion_reply_16(gex_Token_t token,
					 gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
					 gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
					 gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
					 gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14, gex_AM_Arg_t arg15)
 {
    gex_AM_Arg_t args[16];
    args[0] = arg0; args[1] = arg1; args[2] = arg2; args[3] = arg3;
    args[4] = arg4; args[5] = arg5; args[6] = arg6; args[7] = arg7;
    args[8] = arg8; args[9] = arg9; args[10] = arg10; args[11] = arg11;
    args[12] = arg12; args[13] = arg13; args[14] = arg14; args[15] = arg15;
    handle_completion_reply(token, args, 16);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // short requests (arg0 + hdr in rest of args)
  //

  namespace GASNetEXHandlers {

    int send_request_short(gex_EP_t src_ep,
			   gex_Rank_t tgt_rank,
			   gex_EP_Index_t tgt_ep_index,
			   gex_AM_Arg_t arg0,
			   const void *hdr, size_t hdr_bytes,
			   gex_Flags_t flags)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return 1;
#endif

      static const size_t MAX_ARGS = 16;
      gex_AM_Arg_t args[MAX_ARGS];
      args[0] = arg0;
      assert(hdr_bytes <= ((MAX_ARGS - 1) * sizeof(gex_AM_Arg_t)));
      memcpy(args+1, hdr, hdr_bytes);

      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);

      unsigned nargs = 2 + ((hdr_bytes - 1) / sizeof(gex_AM_Arg_t));
      switch(nargs) {
        case 2: return gex_AM_RequestShort2(pair, tgt_rank,
					    HIDX_SHORTREQ(2), flags,
					    args[0], args[1]);
        case 3: return gex_AM_RequestShort3(pair, tgt_rank,
					    HIDX_SHORTREQ(3), flags,
					    args[0], args[1], args[2]);
        case 4: return gex_AM_RequestShort4(pair, tgt_rank,
					    HIDX_SHORTREQ(4), flags,
					    args[0], args[1], args[2], args[3]);
        case 5: return gex_AM_RequestShort5(pair, tgt_rank,
					    HIDX_SHORTREQ(5), flags,
					    args[0], args[1], args[2], args[3],
					    args[4]);
        case 6: return gex_AM_RequestShort6(pair, tgt_rank,
					    HIDX_SHORTREQ(6), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5]);
        case 7: return gex_AM_RequestShort7(pair, tgt_rank,
					    HIDX_SHORTREQ(7), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6]);
        case 8: return gex_AM_RequestShort8(pair, tgt_rank,
					    HIDX_SHORTREQ(8), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7]);
        case 9: return gex_AM_RequestShort9(pair, tgt_rank,
					    HIDX_SHORTREQ(9), flags,
					    args[0], args[1], args[2], args[3],
					    args[4], args[5], args[6], args[7],
					    args[8]);
        case 10: return gex_AM_RequestShort10(pair, tgt_rank,
					      HIDX_SHORTREQ(10), flags,
					      args[0], args[1], args[2], args[3],
					      args[4], args[5], args[6], args[7],
					      args[8], args[9]);
        case 11: return gex_AM_RequestShort11(pair, tgt_rank,
					      HIDX_SHORTREQ(11), flags,
					      args[0], args[1], args[2], args[3],
					      args[4], args[5], args[6], args[7],
					      args[8], args[9], args[10]);
        case 12: return gex_AM_RequestShort12(pair, tgt_rank,
					      HIDX_SHORTREQ(12), flags,
					      args[0], args[1], args[2], args[3],
					      args[4], args[5], args[6], args[7],
					      args[8], args[9], args[10], args[11]);
        case 13: return gex_AM_RequestShort13(pair, tgt_rank,
					      HIDX_SHORTREQ(13), flags,
					      args[0], args[1], args[2], args[3],
					      args[4], args[5], args[6], args[7],
					      args[8], args[9], args[10], args[11],
					      args[12]);
        case 14: return gex_AM_RequestShort14(pair, tgt_rank,
					      HIDX_SHORTREQ(14), flags,
					      args[0], args[1], args[2], args[3],
					      args[4], args[5], args[6], args[7],
					      args[8], args[9], args[10], args[11],
					      args[12], args[13]);
        case 15: return gex_AM_RequestShort15(pair, tgt_rank,
					      HIDX_SHORTREQ(15), flags,
					      args[0], args[1], args[2], args[3],
					      args[4], args[5], args[6], args[7],
					      args[8], args[9], args[10], args[11],
					      args[12], args[13], args[14]);
        case 16: return gex_AM_RequestShort16(pair, tgt_rank,
					      HIDX_SHORTREQ(16), flags,
					      args[0], args[1], args[2], args[3],
					      args[4], args[5], args[6], args[7],
					      args[8], args[9], args[10], args[11],
					      args[12], args[13], args[14], args[15]);
        default: {
	  fprintf(stderr, "ERROR: send_request_short with hdr_bytes=%zd\n", hdr_bytes);
	  abort();
	}
      }
      return 0;
    }

  };

  static void handle_request_short(gex_Token_t token, gex_AM_Arg_t arg0,
				   const void *hdr, size_t hdr_bytes)
  {
    gex_Token_Info_t info;
    // ask for srcrank and ep - both are required, so no need to check result
    gex_Token_Info(token, &info, (GEX_TI_SRCRANK | GEX_TI_EP));

    // ask the ep for the 'internal' pointer
    void *cdata = gex_EP_QueryCData(info.gex_ep);
    GASNetEXInternal *internal = static_cast<GASNetEXInternal *>(cdata);

    gex_AM_Arg_t comp = internal->handle_short(info.gex_srcrank, arg0,
					       hdr, hdr_bytes);
    if(comp != 0)
      gex_AM_ReplyShort1(token, HIDX_COMPREPLY(1), 0/*flags*/, comp);
  }

  // per-arg entry points just assemble header
  static void handle_request_short_2(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1)
  {
    gex_AM_Arg_t hdr[1];
    hdr[0] = arg1;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_3(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2)
  {
    gex_AM_Arg_t hdr[2];
    hdr[0] = arg1; hdr[1] = arg2;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_4(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3)
  {
    gex_AM_Arg_t hdr[3];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_5(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4)
  {
    gex_AM_Arg_t hdr[4];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_6(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5)
  {
    gex_AM_Arg_t hdr[5];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_7(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6)
  {
    gex_AM_Arg_t hdr[6];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_8(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7)
  {
    gex_AM_Arg_t hdr[7];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_9(gex_Token_t token,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8)
  {
    gex_AM_Arg_t hdr[8];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_10(gex_Token_t token,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8, gex_AM_Arg_t arg9)
  {
    gex_AM_Arg_t hdr[9];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_11(gex_Token_t token,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10)
  {
    gex_AM_Arg_t hdr[10];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_12(gex_Token_t token,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11)
  {
    gex_AM_Arg_t hdr[11];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_13(gex_Token_t token,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				      gex_AM_Arg_t arg12)
  {
    gex_AM_Arg_t hdr[12];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_14(gex_Token_t token,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				      gex_AM_Arg_t arg12, gex_AM_Arg_t arg13)
  {
    gex_AM_Arg_t hdr[13];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_15(gex_Token_t token,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				      gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14)
  {
    gex_AM_Arg_t hdr[14];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13; hdr[13] = arg14;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }

  static void handle_request_short_16(gex_Token_t token,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				      gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14, gex_AM_Arg_t arg15)
  {
    gex_AM_Arg_t hdr[15];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13; hdr[13] = arg14; hdr[14] = arg15;
    handle_request_short(token, arg0, hdr, sizeof(hdr));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // medium requests (arg0 + hdr in rest of args)
  //

  namespace GASNetEXHandlers {

    size_t max_request_medium(gex_EP_t src_ep,
			      gex_Rank_t tgt_rank,
			      gex_EP_Index_t tgt_ep_index,
			      size_t hdr_bytes,
			      gex_Event_t *lc_opt, gex_Flags_t flags)
    {
      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);
      unsigned nargs = 2 + ((hdr_bytes - 1) / sizeof(gex_AM_Arg_t));
#if REALM_GEX_RELEASE == 20201100
      // GEX: in 2020.11.0, call below may be a macro that doesn't use
      //  'pair' or 'nargs'
      (void) pair;
      (void) nargs;
#endif
      return gex_AM_MaxRequestMedium(pair, tgt_rank,
				     lc_opt, flags, nargs);
    }

    int send_request_medium(gex_EP_t src_ep,
			    gex_Rank_t tgt_rank,
			    gex_EP_Index_t tgt_ep_index,
			    gex_AM_Arg_t arg0,
			    const void *hdr, size_t hdr_bytes,
			    const void *data, size_t data_bytes,
			    gex_Event_t *lc_opt, gex_Flags_t flags)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return 1;
#endif

      static const size_t MAX_ARGS = 16;
      gex_AM_Arg_t args[MAX_ARGS];
      args[0] = arg0;
      assert(hdr_bytes <= ((MAX_ARGS - 1) * sizeof(gex_AM_Arg_t)));
      memcpy(args+1, hdr, hdr_bytes);

      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);

      // although the docs say requests take const void *'s, it's actually
      //  nonconst?
      void *ncdata = const_cast<void *>(data);

      unsigned nargs = 2 + ((hdr_bytes - 1) / sizeof(gex_AM_Arg_t));
      switch(nargs) {
        case 2: return gex_AM_RequestMedium2(pair, tgt_rank, HIDX_MEDREQ(2),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1]);
        case 3: return gex_AM_RequestMedium3(pair, tgt_rank, HIDX_MEDREQ(3),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1], args[2]);
        case 4: return gex_AM_RequestMedium4(pair, tgt_rank, HIDX_MEDREQ(4),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1], args[2], args[3]);
        case 5: return gex_AM_RequestMedium5(pair, tgt_rank, HIDX_MEDREQ(5),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4]);
        case 6: return gex_AM_RequestMedium6(pair, tgt_rank, HIDX_MEDREQ(6),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5]);
        case 7: return gex_AM_RequestMedium7(pair, tgt_rank, HIDX_MEDREQ(7),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6]);
        case 8: return gex_AM_RequestMedium8(pair, tgt_rank, HIDX_MEDREQ(8),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7]);
        case 9: return gex_AM_RequestMedium9(pair, tgt_rank, HIDX_MEDREQ(9),
					     ncdata, data_bytes, lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8]);
        case 10: return gex_AM_RequestMedium10(pair, tgt_rank, HIDX_MEDREQ(10),
					       ncdata, data_bytes, lc_opt, flags,
					       args[0], args[1], args[2], args[3],
					       args[4], args[5], args[6], args[7],
					       args[8], args[9]);
        case 11: return gex_AM_RequestMedium11(pair, tgt_rank, HIDX_MEDREQ(11),
					       ncdata, data_bytes, lc_opt, flags,
					       args[0], args[1], args[2], args[3],
					       args[4], args[5], args[6], args[7],
					       args[8], args[9], args[10]);
        case 12: return gex_AM_RequestMedium12(pair, tgt_rank, HIDX_MEDREQ(12),
					       ncdata, data_bytes, lc_opt, flags,
					       args[0], args[1], args[2], args[3],
					       args[4], args[5], args[6], args[7],
					       args[8], args[9], args[10], args[11]);
        case 13: return gex_AM_RequestMedium13(pair, tgt_rank, HIDX_MEDREQ(13),
					       ncdata, data_bytes, lc_opt, flags,
					       args[0], args[1], args[2], args[3],
					       args[4], args[5], args[6], args[7],
					       args[8], args[9], args[10], args[11],
					       args[12]);
        case 14: return gex_AM_RequestMedium14(pair, tgt_rank, HIDX_MEDREQ(14),
					       ncdata, data_bytes, lc_opt, flags,
					       args[0], args[1], args[2], args[3],
					       args[4], args[5], args[6], args[7],
					       args[8], args[9], args[10], args[11],
					       args[12], args[13]);
        case 15: return gex_AM_RequestMedium15(pair, tgt_rank, HIDX_MEDREQ(15),
					       ncdata, data_bytes, lc_opt, flags,
					       args[0], args[1], args[2], args[3],
					       args[4], args[5], args[6], args[7],
					       args[8], args[9], args[10], args[11],
					       args[12], args[13], args[14]);
        case 16: return gex_AM_RequestMedium16(pair, tgt_rank, HIDX_MEDREQ(16),
					       ncdata, data_bytes, lc_opt, flags,
					       args[0], args[1], args[2], args[3],
					       args[4], args[5], args[6], args[7],
					       args[8], args[9], args[10], args[11],
					       args[12], args[13], args[14], args[15]);
        default: {
	  fprintf(stderr, "ERROR: send_request_medium with hdr_bytes=%zd\n", hdr_bytes);
	  abort();
	}
      }
      return 0;
    }

    gex_AM_SrcDesc_t prepare_request_medium(gex_EP_t src_ep,
					    gex_Rank_t tgt_rank,
					    gex_EP_Index_t tgt_ep_index,
					    size_t hdr_bytes,
					    const void *data,
					    size_t min_data_bytes,
					    size_t max_data_bytes,
					    gex_Event_t *lc_opt,
					    gex_Flags_t flags)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return GEX_AM_SRCDESC_NO_OP;
#endif

      static const size_t MAX_ARGS = 16;
      assert(hdr_bytes <= ((MAX_ARGS - 1) * sizeof(gex_AM_Arg_t)));

      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);

      // although the docs say requests take const void *'s, it's actually
      //  nonconst?
      void *ncdata = const_cast<void *>(data);

      unsigned nargs = 2 + ((hdr_bytes - 1) / sizeof(gex_AM_Arg_t));

      return gex_AM_PrepareRequestMedium(pair, tgt_rank,
					 ncdata,
					 min_data_bytes, max_data_bytes,
					 lc_opt, flags, nargs);
    }

    void commit_request_medium(gex_AM_SrcDesc_t srcdesc,
			       gex_AM_Arg_t arg0,
			       const void *hdr, size_t hdr_bytes,
			       size_t data_bytes)
    {
      static const size_t MAX_ARGS = 16;
      gex_AM_Arg_t args[MAX_ARGS];
      args[0] = arg0;
      assert(hdr_bytes <= ((MAX_ARGS - 1) * sizeof(gex_AM_Arg_t)));
      memcpy(args+1, hdr, hdr_bytes);

      unsigned nargs = 2 + ((hdr_bytes - 1) / sizeof(gex_AM_Arg_t));
      switch(nargs) {
        case 2:
	  gex_AM_CommitRequestMedium2(srcdesc, HIDX_MEDREQ(2),
				      data_bytes,
				      args[0], args[1]);
	  break;
        case 3:
	  gex_AM_CommitRequestMedium3(srcdesc, HIDX_MEDREQ(3),
				      data_bytes,
				      args[0], args[1], args[2]);
	  break;
        case 4:
	  gex_AM_CommitRequestMedium4(srcdesc, HIDX_MEDREQ(4),
				      data_bytes,
				      args[0], args[1], args[2], args[3]);
	  break;
        case 5:
	  gex_AM_CommitRequestMedium5(srcdesc, HIDX_MEDREQ(5),
				      data_bytes,
				      args[0], args[1], args[2], args[3],
				      args[4]);
	  break;
        case 6:
	  gex_AM_CommitRequestMedium6(srcdesc, HIDX_MEDREQ(6),
				      data_bytes,
				      args[0], args[1], args[2], args[3],
				      args[4], args[5]);
	  break;
        case 7:
	  gex_AM_CommitRequestMedium7(srcdesc, HIDX_MEDREQ(7),
				      data_bytes,
				      args[0], args[1], args[2], args[3],
				      args[4], args[5], args[6]);
	  break;
        case 8:
	  gex_AM_CommitRequestMedium8(srcdesc, HIDX_MEDREQ(8),
				      data_bytes,
				      args[0], args[1], args[2], args[3],
				      args[4], args[5], args[6], args[7]);
	  break;
        case 9:
	  gex_AM_CommitRequestMedium9(srcdesc, HIDX_MEDREQ(9),
				      data_bytes,
				      args[0], args[1], args[2], args[3],
				      args[4], args[5], args[6], args[7],
				      args[8]);
	  break;
        case 10:
	  gex_AM_CommitRequestMedium10(srcdesc, HIDX_MEDREQ(10),
				       data_bytes,
				       args[0], args[1], args[2], args[3],
				       args[4], args[5], args[6], args[7],
				       args[8], args[9]);
	  break;
        case 11:
	  gex_AM_CommitRequestMedium11(srcdesc, HIDX_MEDREQ(11),
				       data_bytes,
				       args[0], args[1], args[2], args[3],
				       args[4], args[5], args[6], args[7],
				       args[8], args[9], args[10]);
	  break;
        case 12:
	  gex_AM_CommitRequestMedium12(srcdesc, HIDX_MEDREQ(12),
				       data_bytes,
				       args[0], args[1], args[2], args[3],
				       args[4], args[5], args[6], args[7],
				       args[8], args[9], args[10], args[11]);
	  break;
        case 13:
	  gex_AM_CommitRequestMedium13(srcdesc, HIDX_MEDREQ(13),
				       data_bytes,
				       args[0], args[1], args[2], args[3],
				       args[4], args[5], args[6], args[7],
				       args[8], args[9], args[10], args[11],
				       args[12]);
	  break;
        case 14:
	  gex_AM_CommitRequestMedium14(srcdesc, HIDX_MEDREQ(14),
				       data_bytes,
				       args[0], args[1], args[2], args[3],
				       args[4], args[5], args[6], args[7],
				       args[8], args[9], args[10], args[11],
				       args[12], args[13]);
	  break;
        case 15:
	  gex_AM_CommitRequestMedium15(srcdesc, HIDX_MEDREQ(15),
				       data_bytes,
				       args[0], args[1], args[2], args[3],
				       args[4], args[5], args[6], args[7],
				       args[8], args[9], args[10], args[11],
				       args[12], args[13], args[14]);
	  break;
        case 16:
	  gex_AM_CommitRequestMedium16(srcdesc, HIDX_MEDREQ(16),
				       data_bytes,
				       args[0], args[1], args[2], args[3],
				       args[4], args[5], args[6], args[7],
				       args[8], args[9], args[10], args[11],
				       args[12], args[13], args[14], args[15]);
	  break;
        default: {
	  fprintf(stderr, "ERROR: commit_request_medium with hdr_bytes=%zd\n", hdr_bytes);
	  abort();
	}
      }
    }
  };

  static void handle_request_medium(gex_Token_t token, gex_AM_Arg_t arg0,
				    const void *hdr, size_t hdr_bytes,
				    const void *data, size_t data_bytes)
  {
    gex_Token_Info_t info;
    // ask for srcrank and ep - both are required, so no need to check result
    gex_Token_Info(token, &info, (GEX_TI_SRCRANK | GEX_TI_EP));

    // ask the ep for the 'internal' pointer
    void *cdata = gex_EP_QueryCData(info.gex_ep);
    GASNetEXInternal *internal = static_cast<GASNetEXInternal *>(cdata);

    gex_AM_Arg_t comp = internal->handle_medium(info.gex_srcrank, arg0,
						hdr, hdr_bytes,
						data, data_bytes);
    if(comp != 0)
      gex_AM_ReplyShort1(token, HIDX_COMPREPLY(1), 0/*flags*/, comp);
  }

  // per-arg entry points just assemble header
  static void handle_request_medium_2(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1)
  {
    gex_AM_Arg_t hdr[1];
    hdr[0] = arg1;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_3(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2)
  {
    gex_AM_Arg_t hdr[2];
    hdr[0] = arg1; hdr[1] = arg2;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_4(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3)
  {
    gex_AM_Arg_t hdr[3];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_5(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4)
  {
    gex_AM_Arg_t hdr[4];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_6(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5)
  {
    gex_AM_Arg_t hdr[5];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_7(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6)
  {
    gex_AM_Arg_t hdr[6];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_8(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7)
  {
    gex_AM_Arg_t hdr[7];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_9(gex_Token_t token,
				      const void *buf, size_t nbytes,
				      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				      gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				      gex_AM_Arg_t arg8)
  {
    gex_AM_Arg_t hdr[8];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_10(gex_Token_t token,
				       const void *buf, size_t nbytes,
				       gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				       gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				       gex_AM_Arg_t arg8, gex_AM_Arg_t arg9)
  {
    gex_AM_Arg_t hdr[9];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_11(gex_Token_t token,
				       const void *buf, size_t nbytes,
				       gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				       gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				       gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10)
  {
    gex_AM_Arg_t hdr[10];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_12(gex_Token_t token,
				       const void *buf, size_t nbytes,
				       gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				       gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				       gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11)
  {
    gex_AM_Arg_t hdr[11];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_13(gex_Token_t token,
				       const void *buf, size_t nbytes,
				       gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				       gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				       gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				       gex_AM_Arg_t arg12)
  {
    gex_AM_Arg_t hdr[12];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_14(gex_Token_t token,
				       const void *buf, size_t nbytes,
				       gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				       gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				       gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				       gex_AM_Arg_t arg12, gex_AM_Arg_t arg13)
  {
    gex_AM_Arg_t hdr[13];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_15(gex_Token_t token,
				       const void *buf, size_t nbytes,
				       gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				       gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				       gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				       gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14)
  {
    gex_AM_Arg_t hdr[14];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13; hdr[13] = arg14;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_medium_16(gex_Token_t token,
				       const void *buf, size_t nbytes,
				       gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				       gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				       gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				       gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14, gex_AM_Arg_t arg15)
  {
    gex_AM_Arg_t hdr[15];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13; hdr[13] = arg14; hdr[14] = arg15;
    handle_request_medium(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // long requests (arg0 + hdr in rest of args)
  //

  namespace GASNetEXHandlers {

    size_t max_request_long(gex_EP_t src_ep,
			    gex_Rank_t tgt_rank,
			    gex_EP_Index_t tgt_ep_index,
			    size_t hdr_bytes,
			    gex_Event_t *lc_opt, gex_Flags_t flags)
    {
      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);
      unsigned nargs = 2 + ((hdr_bytes - 1) / sizeof(gex_AM_Arg_t));
#if REALM_GEX_RELEASE == 20201100
      // GEX: in 2020.11.0, call below may be a macro that doesn't use
      //  'pair' or 'nargs'
      (void) pair;
      (void) nargs;
#endif
      return gex_AM_MaxRequestLong(pair, tgt_rank,
				   lc_opt, flags, nargs);
    }

    int send_request_long(gex_EP_t src_ep,
			  gex_Rank_t tgt_rank,
			  gex_EP_Index_t tgt_ep_index,
			  gex_AM_Arg_t arg0,
			  const void *hdr, size_t hdr_bytes,
			  const void *data, size_t data_bytes,
			  gex_Event_t *lc_opt, gex_Flags_t flags,
			  uintptr_t dest_addr)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return 1;
#endif

      static const size_t MAX_ARGS = 16;
      gex_AM_Arg_t args[MAX_ARGS];
      args[0] = arg0;
      assert(hdr_bytes <= ((MAX_ARGS - 1) * sizeof(gex_AM_Arg_t)));
      memcpy(args+1, hdr, hdr_bytes);

      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);

      // although the docs say requests take const void *'s, it's actually
      //  nonconst?
      void *ncdata = const_cast<void *>(data);

      unsigned nargs = 2 + ((hdr_bytes - 1) / sizeof(gex_AM_Arg_t));
      switch(nargs) {
        case 2: return gex_AM_RequestLong2(pair, tgt_rank, HIDX_LONGREQ(2),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1]);
        case 3: return gex_AM_RequestLong3(pair, tgt_rank, HIDX_LONGREQ(3),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1], args[2]);
        case 4: return gex_AM_RequestLong4(pair, tgt_rank, HIDX_LONGREQ(4),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1], args[2], args[3]);
        case 5: return gex_AM_RequestLong5(pair, tgt_rank, HIDX_LONGREQ(5),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1], args[2], args[3],
					   args[4]);
        case 6: return gex_AM_RequestLong6(pair, tgt_rank, HIDX_LONGREQ(6),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1], args[2], args[3],
					   args[4], args[5]);
        case 7: return gex_AM_RequestLong7(pair, tgt_rank, HIDX_LONGREQ(7),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1], args[2], args[3],
					   args[4], args[5], args[6]);
        case 8: return gex_AM_RequestLong8(pair, tgt_rank, HIDX_LONGREQ(8),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1], args[2], args[3],
					   args[4], args[5], args[6], args[7]);
        case 9: return gex_AM_RequestLong9(pair, tgt_rank, HIDX_LONGREQ(9),
					   ncdata, data_bytes,
					   reinterpret_cast<void *>(dest_addr),
					   lc_opt, flags,
					   args[0], args[1], args[2], args[3],
					   args[4], args[5], args[6], args[7],
					   args[8]);
        case 10: return gex_AM_RequestLong10(pair, tgt_rank, HIDX_LONGREQ(10),
					     ncdata, data_bytes,
					     reinterpret_cast<void *>(dest_addr),
					     lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8], args[9]);
        case 11: return gex_AM_RequestLong11(pair, tgt_rank, HIDX_LONGREQ(11),
					     ncdata, data_bytes,
					     reinterpret_cast<void *>(dest_addr),
					     lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8], args[9], args[10]);
        case 12: return gex_AM_RequestLong12(pair, tgt_rank, HIDX_LONGREQ(12),
					     ncdata, data_bytes,
					     reinterpret_cast<void *>(dest_addr),
					     lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8], args[9], args[10], args[11]);
        case 13: return gex_AM_RequestLong13(pair, tgt_rank, HIDX_LONGREQ(13),
					     ncdata, data_bytes,
					     reinterpret_cast<void *>(dest_addr),
					     lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8], args[9], args[10], args[11],
					     args[12]);
        case 14: return gex_AM_RequestLong14(pair, tgt_rank, HIDX_LONGREQ(14),
					     ncdata, data_bytes,
					     reinterpret_cast<void *>(dest_addr),
					     lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8], args[9], args[10], args[11],
					     args[12], args[13]);
        case 15: return gex_AM_RequestLong15(pair, tgt_rank, HIDX_LONGREQ(15),
					     ncdata, data_bytes,
					     reinterpret_cast<void *>(dest_addr),
					     lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8], args[9], args[10], args[11],
					     args[12], args[13], args[14]);
        case 16: return gex_AM_RequestLong16(pair, tgt_rank, HIDX_LONGREQ(16),
					     ncdata, data_bytes,
					     reinterpret_cast<void *>(dest_addr),
					     lc_opt, flags,
					     args[0], args[1], args[2], args[3],
					     args[4], args[5], args[6], args[7],
					     args[8], args[9], args[10], args[11],
					     args[12], args[13], args[14], args[15]);
        default: {
	  fprintf(stderr, "ERROR: send_request_long with hdr_bytes=%zd\n", hdr_bytes);
	  abort();
	}
      }
      return 0;
    }

  };

  static void handle_request_long(gex_Token_t token, gex_AM_Arg_t arg0,
				  const void *hdr, size_t hdr_bytes,
				  const void *data, size_t data_bytes)
  {
    gex_Token_Info_t info;
    // ask for srcrank and ep - both are required, so no need to check result
    gex_Token_Info(token, &info, (GEX_TI_SRCRANK | GEX_TI_EP));

    // ask the ep for the 'internal' pointer
    void *cdata = gex_EP_QueryCData(info.gex_ep);
    GASNetEXInternal *internal = static_cast<GASNetEXInternal *>(cdata);

    gex_AM_Arg_t comp = internal->handle_long(info.gex_srcrank, arg0,
					      hdr, hdr_bytes,
					      data, data_bytes);
    if(comp != 0)
      gex_AM_ReplyShort1(token, HIDX_COMPREPLY(1), 0/*flags*/, comp);
  }

  // per-arg entry points just assemble header
  static void handle_request_long_2(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1)
  {
    gex_AM_Arg_t hdr[1];
    hdr[0] = arg1;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_3(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2)
  {
    gex_AM_Arg_t hdr[2];
    hdr[0] = arg1; hdr[1] = arg2;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_4(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3)
  {
    gex_AM_Arg_t hdr[3];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_5(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				    gex_AM_Arg_t arg4)
  {
    gex_AM_Arg_t hdr[4];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_6(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				    gex_AM_Arg_t arg4, gex_AM_Arg_t arg5)
  {
    gex_AM_Arg_t hdr[5];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_7(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				    gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6)
  {
    gex_AM_Arg_t hdr[6];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_8(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				    gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7)
  {
    gex_AM_Arg_t hdr[7];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_9(gex_Token_t token,
				    const void *buf, size_t nbytes,
				    gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				    gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				    gex_AM_Arg_t arg8)
  {
    gex_AM_Arg_t hdr[8];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_10(gex_Token_t token,
				     const void *buf, size_t nbytes,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8, gex_AM_Arg_t arg9)
  {
    gex_AM_Arg_t hdr[9];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_11(gex_Token_t token,
				     const void *buf, size_t nbytes,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10)
  {
    gex_AM_Arg_t hdr[10];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_12(gex_Token_t token,
				     const void *buf, size_t nbytes,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11)
  {
    gex_AM_Arg_t hdr[11];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_13(gex_Token_t token,
				     const void *buf, size_t nbytes,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				     gex_AM_Arg_t arg12)
  {
    gex_AM_Arg_t hdr[12];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_14(gex_Token_t token,
				     const void *buf, size_t nbytes,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				     gex_AM_Arg_t arg12, gex_AM_Arg_t arg13)
  {
    gex_AM_Arg_t hdr[13];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_15(gex_Token_t token,
				     const void *buf, size_t nbytes,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				     gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14)
  {
    gex_AM_Arg_t hdr[14];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13; hdr[13] = arg14;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }

  static void handle_request_long_16(gex_Token_t token,
				     const void *buf, size_t nbytes,
				     gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				     gex_AM_Arg_t arg4, gex_AM_Arg_t arg5, gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				     gex_AM_Arg_t arg8, gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
				     gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14, gex_AM_Arg_t arg15)
  {
    gex_AM_Arg_t hdr[15];
    hdr[0] = arg1; hdr[1] = arg2; hdr[2] = arg3; hdr[3] = arg4;
    hdr[4] = arg5; hdr[5] = arg6; hdr[6] = arg7; hdr[7] = arg8;
    hdr[8] = arg9; hdr[9] = arg10; hdr[10] = arg11; hdr[11] = arg12;
    hdr[12] = arg13; hdr[13] = arg14; hdr[14] = arg15;
    handle_request_long(token, arg0, hdr, sizeof(hdr), buf, nbytes);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // reverse get
  //

  namespace GASNetEXHandlers {

    // sends a long as a "reverse get": the _header_ is sent as the
    //  payload of a medium that contains the src ep_index/ptr so that
    //  the target can do an RMA get of the actual payload
    int send_request_rget(gex_TM_t prim_tm,
			  gex_Rank_t tgt_rank,
			  gex_EP_Index_t tgt_ep_index,
			  gex_AM_Arg_t arg0,
			  const void *hdr, size_t hdr_bytes,
			  gex_EP_Index_t payload_ep_index,
			  const void *payload, size_t payload_bytes,
			  gex_Event_t *lc_opt, gex_Flags_t flags,
			  uintptr_t dest_addr)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return 1;
#endif

      return gex_AM_RequestMedium9(prim_tm, tgt_rank,
				   HIDX_LONG_AS_GET,
				   const_cast<void *>(hdr), hdr_bytes,
				   lc_opt,
				   flags,
				   arg0,
				   payload_ep_index,
				   tgt_ep_index,
				   ARG_LO(payload),
				   ARG_HI(payload),
				   ARG_LO(payload_bytes),
				   ARG_HI(payload_bytes),
				   ARG_LO(dest_addr),
				   ARG_HI(dest_addr));
    }

  }

  static void handle_request_rget(gex_Token_t token,
				  const void *buf, size_t nbytes,
				  gex_AM_Arg_t arg0, gex_AM_Arg_t arg1,
				  gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
				  gex_AM_Arg_t arg4, gex_AM_Arg_t arg5,
				  gex_AM_Arg_t arg6, gex_AM_Arg_t arg7,
				  gex_AM_Arg_t arg8)
  {
    gex_Token_Info_t info;
    // ask for srcrank and ep - both are required, so no need to check result
    gex_Token_Info(token, &info, (GEX_TI_SRCRANK | GEX_TI_EP));

    // ask the ep for the 'internal' pointer
    void *cdata = gex_EP_QueryCData(info.gex_ep);
    GASNetEXInternal *internal = static_cast<GASNetEXInternal *>(cdata);

    gex_EP_Index_t payload_ep_index = arg1;
    gex_EP_Index_t tgt_ep_index = arg2;
    uintptr_t src_ptr = ARG_COMBINE(arg3, arg4);
    size_t src_size = ARG_COMBINE(arg5, arg6);
    uintptr_t tgt_ptr = ARG_COMBINE(arg7, arg8);

    internal->handle_reverse_get(info.gex_srcrank, payload_ep_index,
				 tgt_ep_index, arg0,
				 buf, nbytes,
				 src_ptr, tgt_ptr, src_size);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // put header
  //

  namespace GASNetEXHandlers {

    // send the header of a long after the payload has been delivered via
    //  an RMA put: header is sent as a medium
    int send_request_put_header(gex_EP_t src_ep,
                                gex_Rank_t tgt_rank,
                                gex_EP_Index_t tgt_ep_index,
                                gex_AM_Arg_t arg0,
                                const void *hdr, size_t hdr_bytes,
                                uintptr_t dest_addr, size_t payload_bytes,
                                gex_Event_t *lc_opt, gex_Flags_t flags)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return 1;
#endif

      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);

      return gex_AM_RequestMedium5(pair, tgt_rank,
				   HIDX_PUT_HEADER,
				   const_cast<void *>(hdr), hdr_bytes,
				   lc_opt,
				   flags,
				   arg0,
                                   ARG_LO(dest_addr),
                                   ARG_HI(dest_addr),
				   ARG_LO(payload_bytes),
				   ARG_HI(payload_bytes));
    }

  }

  static void handle_request_put_header(gex_Token_t token,
                                        const void *buf, size_t nbytes,
                                        gex_AM_Arg_t arg0, gex_AM_Arg_t arg1,
                                        gex_AM_Arg_t arg2, gex_AM_Arg_t arg3,
                                        gex_AM_Arg_t arg4)
  {
    gex_Token_Info_t info;
    // ask for srcrank and ep - both are required, so no need to check result
    gex_Token_Info(token, &info, (GEX_TI_SRCRANK | GEX_TI_EP));

    // ask the ep for the 'internal' pointer
    void *cdata = gex_EP_QueryCData(info.gex_ep);
    GASNetEXInternal *internal = static_cast<GASNetEXInternal *>(cdata);

    uintptr_t payload_ptr = ARG_COMBINE(arg1, arg2);
    size_t payload_bytes = ARG_COMBINE(arg3, arg4);

    gex_AM_Arg_t comp = internal->handle_long(info.gex_srcrank, arg0,
                                              buf, nbytes,
                                              reinterpret_cast<const void *>(payload_ptr),
                                              payload_bytes);
    if(comp != 0)
      gex_AM_ReplyShort1(token, HIDX_COMPREPLY(1), 0/*flags*/, comp);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // batched requests
  //

  namespace GASNetEXHandlers {

    gex_AM_SrcDesc_t prepare_request_batch(gex_EP_t src_ep,
					   gex_Rank_t tgt_rank,
					   gex_EP_Index_t tgt_ep_index,
					   const void *data,
					   size_t min_data_bytes,
					   size_t max_data_bytes,
					   gex_Event_t *lc_opt,
					   gex_Flags_t flags)
    {
#ifdef DEBUG_REALM
      if(((flags & GEX_FLAG_IMMEDIATE) != 0) &&
	 check_artificial_backpressure())
	return GEX_AM_SRCDESC_NO_OP;
#endif

      gex_TM_t pair = gex_TM_Pair(src_ep, tgt_ep_index);

      // although the docs say requests take const void *'s, it's actually
      //  nonconst?
      void *ncdata = const_cast<void *>(data);

      unsigned nargs = 2;

      return gex_AM_PrepareRequestMedium(pair, tgt_rank,
					 ncdata,
					 min_data_bytes, max_data_bytes,
					 lc_opt, flags, nargs);
    }

    void commit_request_batch(gex_AM_SrcDesc_t srcdesc,
			      gex_AM_Arg_t arg0, gex_AM_Arg_t cksum,
			      size_t data_bytes)
    {
      gex_AM_CommitRequestMedium2(srcdesc, HIDX_BATCHREQ,
				  data_bytes, arg0, cksum);
    }

  };

  void handle_request_batch(gex_Token_t token,
			    const void *buf, size_t nbytes,
			    gex_AM_Arg_t arg0, gex_AM_Arg_t cksum)
  {
    gex_Token_Info_t info;
    // ask for srcrank and ep - both are required, so no need to check result
    gex_Token_Info(token, &info, (GEX_TI_SRCRANK | GEX_TI_EP));

    // ask the ep for the 'internal' pointer
    void *cdata = gex_EP_QueryCData(info.gex_ep);
    GASNetEXInternal *internal = static_cast<GASNetEXInternal *>(cdata);

    static const int MAX_BATCH_SIZE = 16;
    assert(arg0 <= MAX_BATCH_SIZE);
    gex_AM_Arg_t comps[MAX_BATCH_SIZE];

    size_t ncomp = internal->handle_batch(info.gex_srcrank, arg0, cksum,
					  buf, nbytes, comps);

    switch(ncomp) {
    case 0:
      // no reply needed
      break;
    case 1:
      gex_AM_ReplyShort1(token, HIDX_COMPREPLY(1), 0/*flags*/,
			 comps[0]);
      break;
    case 2:
      gex_AM_ReplyShort2(token, HIDX_COMPREPLY(2), 0/*flags*/,
			 comps[0], comps[1]);
      break;
    case 3:
      gex_AM_ReplyShort3(token, HIDX_COMPREPLY(3), 0/*flags*/,
			 comps[0], comps[1], comps[2]);
      break;
    case 4:
      gex_AM_ReplyShort4(token, HIDX_COMPREPLY(4), 0/*flags*/,
			 comps[0], comps[1], comps[2], comps[3]);
      break;
    case 5:
      gex_AM_ReplyShort5(token, HIDX_COMPREPLY(5), 0/*flags*/,
			 comps[0], comps[1], comps[2], comps[3],
			 comps[4]);
      break;
    case 6:
      gex_AM_ReplyShort6(token, HIDX_COMPREPLY(6), 0/*flags*/,
			 comps[0], comps[1], comps[2], comps[3],
			 comps[4], comps[5]);
      break;
    case 7:
      gex_AM_ReplyShort7(token, HIDX_COMPREPLY(7), 0/*flags*/,
			 comps[0], comps[1], comps[2], comps[3],
			 comps[4], comps[5], comps[6]);
      break;
    case 8:
      gex_AM_ReplyShort8(token, HIDX_COMPREPLY(8), 0/*flags*/,
			 comps[0], comps[1], comps[2], comps[3],
			 comps[4], comps[5], comps[6], comps[7]);
      break;
    case 9:
      gex_AM_ReplyShort9(token, HIDX_COMPREPLY(9), 0/*flags*/,
			 comps[0], comps[1], comps[2], comps[3],
			 comps[4], comps[5], comps[6], comps[7],
			 comps[8]);
      break;
    case 10:
      gex_AM_ReplyShort10(token, HIDX_COMPREPLY(10), 0/*flags*/,
			  comps[0], comps[1], comps[2], comps[3],
			  comps[4], comps[5], comps[6], comps[7],
			  comps[8], comps[9]);
      break;
    case 11:
      gex_AM_ReplyShort11(token, HIDX_COMPREPLY(11), 0/*flags*/,
			  comps[0], comps[1], comps[2], comps[3],
			  comps[4], comps[5], comps[6], comps[7],
			  comps[8], comps[9], comps[10]);
      break;
    case 12:
      gex_AM_ReplyShort12(token, HIDX_COMPREPLY(12), 0/*flags*/,
			  comps[0], comps[1], comps[2], comps[3],
			  comps[4], comps[5], comps[6], comps[7],
			  comps[8], comps[9], comps[10], comps[11]);
      break;
    case 13:
      gex_AM_ReplyShort13(token, HIDX_COMPREPLY(13), 0/*flags*/,
			  comps[0], comps[1], comps[2], comps[3],
			  comps[4], comps[5], comps[6], comps[7],
			  comps[8], comps[9], comps[10], comps[11],
			  comps[12]);
      break;
    case 14:
      gex_AM_ReplyShort14(token, HIDX_COMPREPLY(14), 0/*flags*/,
			  comps[0], comps[1], comps[2], comps[3],
			  comps[4], comps[5], comps[6], comps[7],
			  comps[8], comps[9], comps[10], comps[11],
			  comps[12], comps[13]);
      break;
    case 15:
      gex_AM_ReplyShort15(token, HIDX_COMPREPLY(15), 0/*flags*/,
			  comps[0], comps[1], comps[2], comps[3],
			  comps[4], comps[5], comps[6], comps[7],
			  comps[8], comps[9], comps[10], comps[11],
			  comps[12], comps[13], comps[14]);
      break;
    case 16:
      gex_AM_ReplyShort16(token, HIDX_COMPREPLY(16), 0/*flags*/,
			  comps[0], comps[1], comps[2], comps[3],
			  comps[4], comps[5], comps[6], comps[7],
			  comps[8], comps[9], comps[10], comps[11],
			  comps[12], comps[13], comps[14], comps[15]);
      break;
    default:
      fprintf(stderr, "ERROR: handle_request_batch reply with ncomp=%zd\n", ncomp);
      abort();
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // handler table
  //

  namespace GASNetEXHandlers {

#define REALM_GEX_COMPREPLY_HANDLER(narg, fname)				\
  { HIDX_COMPREPLY(narg), reinterpret_cast<void(*)()>(fname), GEX_FLAG_AM_REQREP|GEX_FLAG_AM_SHORT, narg, nullptr, #fname }
#define REALM_GEX_SHORTREQ_HANDLER(narg, fname)				\
  { HIDX_SHORTREQ(narg), reinterpret_cast<void(*)()>(fname), GEX_FLAG_AM_REQUEST|GEX_FLAG_AM_SHORT, narg, nullptr, #fname }
#define REALM_GEX_MEDREQ_HANDLER(narg, fname)					\
  { HIDX_MEDREQ(narg), reinterpret_cast<void(*)()>(fname), GEX_FLAG_AM_REQUEST|GEX_FLAG_AM_MEDIUM, narg, nullptr, #fname }
#define REALM_GEX_LONGREQ_HANDLER(narg, fname)				\
  { HIDX_LONGREQ(narg), reinterpret_cast<void(*)()>(fname), GEX_FLAG_AM_REQUEST|GEX_FLAG_AM_LONG, narg, nullptr, #fname }

    gex_AM_Entry_t handler_table[] = {
      REALM_GEX_SHORTREQ_HANDLER(2, handle_request_short_2),
      REALM_GEX_SHORTREQ_HANDLER(3, handle_request_short_3),
      REALM_GEX_SHORTREQ_HANDLER(4, handle_request_short_4),
      REALM_GEX_SHORTREQ_HANDLER(5, handle_request_short_5),
      REALM_GEX_SHORTREQ_HANDLER(6, handle_request_short_6),
      REALM_GEX_SHORTREQ_HANDLER(7, handle_request_short_7),
      REALM_GEX_SHORTREQ_HANDLER(8, handle_request_short_8),
      REALM_GEX_SHORTREQ_HANDLER(9, handle_request_short_9),
      REALM_GEX_SHORTREQ_HANDLER(10, handle_request_short_10),
      REALM_GEX_SHORTREQ_HANDLER(11, handle_request_short_11),
      REALM_GEX_SHORTREQ_HANDLER(12, handle_request_short_12),
      REALM_GEX_SHORTREQ_HANDLER(13, handle_request_short_13),
      REALM_GEX_SHORTREQ_HANDLER(14, handle_request_short_14),
      REALM_GEX_SHORTREQ_HANDLER(15, handle_request_short_15),
      REALM_GEX_SHORTREQ_HANDLER(16, handle_request_short_16),

      REALM_GEX_MEDREQ_HANDLER(2, handle_request_medium_2),
      REALM_GEX_MEDREQ_HANDLER(3, handle_request_medium_3),
      REALM_GEX_MEDREQ_HANDLER(4, handle_request_medium_4),
      REALM_GEX_MEDREQ_HANDLER(5, handle_request_medium_5),
      REALM_GEX_MEDREQ_HANDLER(6, handle_request_medium_6),
      REALM_GEX_MEDREQ_HANDLER(7, handle_request_medium_7),
      REALM_GEX_MEDREQ_HANDLER(8, handle_request_medium_8),
      REALM_GEX_MEDREQ_HANDLER(9, handle_request_medium_9),
      REALM_GEX_MEDREQ_HANDLER(10, handle_request_medium_10),
      REALM_GEX_MEDREQ_HANDLER(11, handle_request_medium_11),
      REALM_GEX_MEDREQ_HANDLER(12, handle_request_medium_12),
      REALM_GEX_MEDREQ_HANDLER(13, handle_request_medium_13),
      REALM_GEX_MEDREQ_HANDLER(14, handle_request_medium_14),
      REALM_GEX_MEDREQ_HANDLER(15, handle_request_medium_15),
      REALM_GEX_MEDREQ_HANDLER(16, handle_request_medium_16),

      REALM_GEX_LONGREQ_HANDLER(2, handle_request_long_2),
      REALM_GEX_LONGREQ_HANDLER(3, handle_request_long_3),
      REALM_GEX_LONGREQ_HANDLER(4, handle_request_long_4),
      REALM_GEX_LONGREQ_HANDLER(5, handle_request_long_5),
      REALM_GEX_LONGREQ_HANDLER(6, handle_request_long_6),
      REALM_GEX_LONGREQ_HANDLER(7, handle_request_long_7),
      REALM_GEX_LONGREQ_HANDLER(8, handle_request_long_8),
      REALM_GEX_LONGREQ_HANDLER(9, handle_request_long_9),
      REALM_GEX_LONGREQ_HANDLER(10, handle_request_long_10),
      REALM_GEX_LONGREQ_HANDLER(11, handle_request_long_11),
      REALM_GEX_LONGREQ_HANDLER(12, handle_request_long_12),
      REALM_GEX_LONGREQ_HANDLER(13, handle_request_long_13),
      REALM_GEX_LONGREQ_HANDLER(14, handle_request_long_14),
      REALM_GEX_LONGREQ_HANDLER(15, handle_request_long_15),
      REALM_GEX_LONGREQ_HANDLER(16, handle_request_long_16),

      REALM_GEX_COMPREPLY_HANDLER(1, handle_completion_reply_1),
      REALM_GEX_COMPREPLY_HANDLER(2, handle_completion_reply_2),
      REALM_GEX_COMPREPLY_HANDLER(3, handle_completion_reply_3),
      REALM_GEX_COMPREPLY_HANDLER(4, handle_completion_reply_4),
      REALM_GEX_COMPREPLY_HANDLER(5, handle_completion_reply_5),
      REALM_GEX_COMPREPLY_HANDLER(6, handle_completion_reply_6),
      REALM_GEX_COMPREPLY_HANDLER(7, handle_completion_reply_7),
      REALM_GEX_COMPREPLY_HANDLER(8, handle_completion_reply_8),
      REALM_GEX_COMPREPLY_HANDLER(9, handle_completion_reply_9),
      REALM_GEX_COMPREPLY_HANDLER(10, handle_completion_reply_10),
      REALM_GEX_COMPREPLY_HANDLER(11, handle_completion_reply_11),
      REALM_GEX_COMPREPLY_HANDLER(12, handle_completion_reply_12),
      REALM_GEX_COMPREPLY_HANDLER(13, handle_completion_reply_13),
      REALM_GEX_COMPREPLY_HANDLER(14, handle_completion_reply_14),
      REALM_GEX_COMPREPLY_HANDLER(15, handle_completion_reply_15),
      REALM_GEX_COMPREPLY_HANDLER(16, handle_completion_reply_16),

      { HIDX_LONG_AS_GET,
	reinterpret_cast<void(*)()>(handle_request_rget),
	GEX_FLAG_AM_REQUEST|GEX_FLAG_AM_MEDIUM, 9,
	nullptr, "handle_request_rget" },

      { HIDX_PUT_HEADER,
	reinterpret_cast<void(*)()>(handle_request_put_header),
	GEX_FLAG_AM_REQUEST|GEX_FLAG_AM_MEDIUM, 5,
	nullptr, "handle_request_put_header" },

      { HIDX_BATCHREQ,
	reinterpret_cast<void(*)()>(handle_request_batch),
	GEX_FLAG_AM_REQUEST|GEX_FLAG_AM_MEDIUM, 2,
	nullptr, "handle_request_batch" }
    };

    size_t handler_table_size = sizeof(handler_table) / sizeof(handler_table[0]);

  };


}; // namespace Realm
