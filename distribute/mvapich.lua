------------------------------------------------------------------------------
-- MPI for Lua.
-- Copyright © 2013–2015 Peter Colberg.
-- Distributed under the MIT license. (See accompanying file LICENSE.)
------------------------------------------------------------------------------

local ffi = require("ffi")

ffi.cdef[[
typedef int MPI_Comm;
typedef int MPI_Op;
typedef int MPI_Group;
typedef int MPI_Datatype;
typedef int MPI_Request;
typedef int MPI_Errhandler;
typedef int MPI_Message;

typedef struct _MPI_Comm _MPI_Comm;
struct _MPI_Comm {
    MPI_Comm common_world;
};

typedef struct _MPI_Datatype _MPI_Datatype;
struct _MPI_Datatype {
     MPI_Datatype datatype;
};

/* Define some null objects */
static const int MPI_COMM_NULL     = ((MPI_Comm)0x04000000);
static const int MPI_OP_NULL       = ((MPI_Op)0x18000000);
static const int MPI_GROUP_NULL    = ((MPI_Group)0x08000000);
static const int MPI_DATATYPE_NULL = ((MPI_Datatype)0x0c000000);
static const int MPI_REQUEST_NULL  = ((MPI_Request)0x2c000000);
static const int MPI_ERRHANDLER_NULL= ((MPI_Errhandler)0x14000000);
static const int MPI_MESSAGE_NULL  = ((MPI_Message)MPI_REQUEST_NULL);
static const int MPI_MESSAGE_NO_PROC= ((MPI_Message)0x6c000000);

static const int MPI_IDENT    = 0;
static const int MPI_CONGRUENT= 1;
static const int MPI_SIMILAR  = 2;
static const int MPI_UNEQUAL  = 3;

static const int MPI_CHAR          = ((MPI_Datatype)0x4c000101);
static const int MPI_SIGNED_CHAR   = ((MPI_Datatype)0x4c000118);
static const int MPI_UNSIGNED_CHAR = ((MPI_Datatype)0x4c000102);
static const int MPI_BYTE          = ((MPI_Datatype)0x4c00010d);
static const int MPI_WCHAR         = ((MPI_Datatype)0x4c00040e);
static const int MPI_SHORT         = ((MPI_Datatype)0x4c000203);
static const int MPI_UNSIGNED_SHORT= ((MPI_Datatype)0x4c000204);
static const int MPI_INT           = ((MPI_Datatype)0x4c000405);
static const int MPI_UNSIGNED      = ((MPI_Datatype)0x4c000406);
static const int MPI_LONG          = ((MPI_Datatype)0x4c000807);
static const int MPI_UNSIGNED_LONG = ((MPI_Datatype)0x4c000808);
static const int MPI_FLOAT         = ((MPI_Datatype)0x4c00040a);
static const int MPI_DOUBLE        = ((MPI_Datatype)0x4c00080b);
static const int MPI_LONG_DOUBLE   = ((MPI_Datatype)0x4c00100c);
static const int MPI_LONG_LONG_INT = ((MPI_Datatype)0x4c000809);
static const int MPI_UNSIGNED_LONG_LONG= ((MPI_Datatype)0x4c000819);
static const int MPI_LONG_LONG     = MPI_LONG_LONG_INT;

static const int MPI_PACKED        = ((MPI_Datatype)0x4c00010f);
static const int MPI_LB            = ((MPI_Datatype)0x4c000010);
static const int MPI_UB            = ((MPI_Datatype)0x4c000011);

/* 
   The layouts for the types MPI_DOUBLE_INT etc are simply
   struct { 
       double var;
       int    loc;
   }
   This is documented in the man pages on the various datatypes.   
 */
static const int MPI_FLOAT_INT        = ((MPI_Datatype)0x8c000000);
static const int MPI_DOUBLE_INT       = ((MPI_Datatype)0x8c000001);
static const int MPI_LONG_INT         = ((MPI_Datatype)0x8c000002);
static const int MPI_SHORT_INT        = ((MPI_Datatype)0x8c000003);
static const int MPI_2INT             = ((MPI_Datatype)0x4c000816);
static const int MPI_LONG_DOUBLE_INT  = ((MPI_Datatype)0x8c000004);

/* Fortran types */
static const int MPI_COMPLEX          = ((MPI_Datatype)1275070494);
static const int MPI_DOUBLE_COMPLEX   = ((MPI_Datatype)1275072546);
static const int MPI_LOGICAL          = ((MPI_Datatype)1275069469);
static const int MPI_REAL             = ((MPI_Datatype)1275069468);
static const int MPI_DOUBLE_PRECISION = ((MPI_Datatype)1275070495);
static const int MPI_INTEGER          = ((MPI_Datatype)1275069467);
static const int MPI_2INTEGER         = ((MPI_Datatype)1275070496);
static const int MPI_2REAL            = ((MPI_Datatype)1275070497);
static const int MPI_2DOUBLE_PRECISION= ((MPI_Datatype)1275072547);
static const int MPI_CHARACTER        = ((MPI_Datatype)1275068698);

/* Size-specific types (see MPI-2, 10.2.5) */
static const int MPI_REAL4            = ((MPI_Datatype)0x4c000427);
static const int MPI_REAL8            = ((MPI_Datatype)0x4c000829);
static const int MPI_REAL16           = ((MPI_Datatype)0x4c00102b);
static const int MPI_COMPLEX8         = ((MPI_Datatype)0x4c000828);
static const int MPI_COMPLEX16        = ((MPI_Datatype)0x4c00102a);
static const int MPI_COMPLEX32        = ((MPI_Datatype)0x4c00202c);
static const int MPI_INTEGER1         = ((MPI_Datatype)0x4c00012d);
static const int MPI_INTEGER2         = ((MPI_Datatype)0x4c00022f);
static const int MPI_INTEGER4         = ((MPI_Datatype)0x4c000430);
static const int MPI_INTEGER8         = ((MPI_Datatype)0x4c000831);
static const int MPI_INTEGER16        = ((MPI_Datatype)MPI_DATATYPE_NULL);

/* C99 fixed-width datatypes */
static const int MPI_INT8_T           = ((MPI_Datatype)0x4c000137);
static const int MPI_INT16_T          = ((MPI_Datatype)0x4c000238);
static const int MPI_INT32_T          = ((MPI_Datatype)0x4c000439);
static const int MPI_INT64_T          = ((MPI_Datatype)0x4c00083a);
static const int MPI_UINT8_T          = ((MPI_Datatype)0x4c00013b);
static const int MPI_UINT16_T         = ((MPI_Datatype)0x4c00023c);
static const int MPI_UINT32_T         = ((MPI_Datatype)0x4c00043d);
static const int MPI_UINT64_T         = ((MPI_Datatype)0x4c00083e);

/* other C99 types */
static const int MPI_C_BOOL                = ((MPI_Datatype)0x4c00013f);
static const int MPI_C_FLOAT_COMPLEX       = ((MPI_Datatype)0x4c000840);
static const int MPI_C_COMPLEX             = MPI_C_FLOAT_COMPLEX;
static const int MPI_C_DOUBLE_COMPLEX      = ((MPI_Datatype)0x4c001041);
static const int MPI_C_LONG_DOUBLE_COMPLEX = ((MPI_Datatype)0x4c002042);

/* address/offset types */
static const int MPI_AINT         = ((MPI_Datatype)0x4c000843);
static const int MPI_OFFSET       = ((MPI_Datatype)0x4c000844);
static const int MPI_COUNT        = ((MPI_Datatype)0x4c000845);

/* MPI-3 C++ types */
static const int MPI_CXX_BOOL               = ((MPI_Datatype)0x4c000133);
static const int MPI_CXX_FLOAT_COMPLEX      = ((MPI_Datatype)0x4c000834);
static const int MPI_CXX_DOUBLE_COMPLEX     = ((MPI_Datatype)0x4c001035);
static const int MPI_CXX_LONG_DOUBLE_COMPLEX= ((MPI_Datatype)0x4c002036);

/* typeclasses */
static const int MPI_TYPECLASS_REAL = 1;
static const int MPI_TYPECLASS_INTEGER= 2;
static const int MPI_TYPECLASS_COMPLEX= 3;

/* Communicators */
//static const int MPI_COMM_WORLD= ((MPI_Comm)0x44000000);
static const int MPI_COMM_WORLD= 0x44000000;
static const int MPI_COMM_SELF = ((MPI_Comm)0x44000001);

/* Groups */
static const int MPI_GROUP_EMPTY= ((MPI_Group)0x48000000);

/* RMA and Windows */
typedef int MPI_Win;
static const int MPI_WIN_NULL= ((MPI_Win)0x20000000);

/* File and IO */
/* This define lets ROMIO know that MPI_File has been defined */
//static const int MPI_FILE_DEFINED
/* ROMIO uses a pointer for MPI_File objects.  This must be the same definition
   as in src/mpi/romio/include/mpio.h.in  */
typedef struct ADIOI_FileD *MPI_File;
static const int MPI_FILE_NULL= ((MPI_File)0);

/* Collective operations */

static const int MPI_MAX    = (MPI_Op)(0x58000001);
static const int MPI_MIN    = (MPI_Op)(0x58000002);
static const int MPI_SUM    = (MPI_Op)(0x58000003);
static const int MPI_PROD   = (MPI_Op)(0x58000004);
static const int MPI_LAND   = (MPI_Op)(0x58000005);
static const int MPI_BAND   = (MPI_Op)(0x58000006);
static const int MPI_LOR    = (MPI_Op)(0x58000007);
static const int MPI_BOR    = (MPI_Op)(0x58000008);
static const int MPI_LXOR   = (MPI_Op)(0x58000009);
static const int MPI_BXOR   = (MPI_Op)(0x5800000a);
static const int MPI_MINLOC = (MPI_Op)(0x5800000b);
static const int MPI_MAXLOC = (MPI_Op)(0x5800000c);
static const int MPI_REPLACE= (MPI_Op)(0x5800000d);
static const int MPI_NO_OP  = (MPI_Op)(0x5800000e);

/* Permanent key values */
/* C Versions (return pointer to value),
   Fortran Versions (return integer value).
   Handled directly by the attribute value routine
   
   DO NOT CHANGE THESE.  The values encode:
   builtin kind (0x1 in bit 30-31)
   Keyval object (0x9 in bits 26-29)
   for communicator (0x1 in bits 22-25)
   
   Fortran versions of the attributes are formed by adding one to
   the C version.
 */
static const int MPI_TAG_UB          = 0x64400001;
static const int MPI_HOST            = 0x64400003;
static const int MPI_IO              = 0x64400005;
static const int MPI_WTIME_IS_GLOBAL = 0x64400007;
static const int MPI_UNIVERSE_SIZE   = 0x64400009;
static const int MPI_LASTUSEDCODE    = 0x6440000b;
static const int MPI_APPNUM          = 0x6440000d;

/* In addition, there are 5 predefined window attributes that are
   defined for every window */
static const int MPI_WIN_BASE         = 0x66000001;
static const int MPI_WIN_SIZE         = 0x66000003;
static const int MPI_WIN_DISP_UNIT    = 0x66000005;
static const int MPI_WIN_CREATE_FLAVOR= 0x66000007;
static const int MPI_WIN_MODEL        = 0x66000009;

/* These are only guesses; make sure you change them in mpif.h as well */
static const int MPI_MAX_PROCESSOR_NAME= 128;
static const int MPI_MAX_LIBRARY_VERSION_STRING= 8192;
static const int MPI_MAX_ERROR_STRING  = 512;
static const int MPI_MAX_PORT_NAME     = 256;
static const int MPI_MAX_OBJECT_NAME   = 128;

/* Pre-defined constants */
static const int MPI_UNDEFINED     = -32766;
static const int MPI_KEYVAL_INVALID = 0x24000000;

/* Upper bound on the overhead in bsend for each message buffer */
static const int MPI_BSEND_OVERHEAD= 96;

/* Topology types */
typedef enum MPIR_Topo_type { MPI_GRAPH=1, MPI_CART=2, MPI_DIST_GRAPH=3 } MPIR_Topo_type;

//static const char* MPI_BOTTOM     = (char*)0;
static const int MPI_BOTTOM     = 0;
extern int * const MPI_UNWEIGHTED;
extern int * const MPI_WEIGHTS_EMPTY;

static const int MPI_PROC_NULL  = (-1);
static const int MPI_ANY_SOURCE = (-2);
static const int MPI_ROOT       = (-3);
static const int MPI_ANY_TAG    = (-1);

static const int MPI_LOCK_EXCLUSIVE = 234;
static const int MPI_LOCK_SHARED    = 235;
/* C functions */
typedef void (MPI_Handler_function) ( MPI_Comm *, int *, ... );
typedef int (MPI_Comm_copy_attr_function)(MPI_Comm, int, void *, void *,
                                          void *, int *);
typedef int (MPI_Comm_delete_attr_function)(MPI_Comm, int, void *, void *);
typedef int (MPI_Type_copy_attr_function)(MPI_Datatype, int, void *, void *,
                                          void *, int *);
typedef int (MPI_Type_delete_attr_function)(MPI_Datatype, int, void *, void *);
typedef int (MPI_Win_copy_attr_function)(MPI_Win, int, void *, void *, void *,
                                         int *);
typedef int (MPI_Win_delete_attr_function)(MPI_Win, int, void *, void *);
/* added in MPI-2.2 */
typedef void (MPI_Comm_errhandler_function)(MPI_Comm *, int *, ...);
typedef void (MPI_File_errhandler_function)(MPI_File *, int *, ...);
typedef void (MPI_Win_errhandler_function)(MPI_Win *, int *, ...);
/* names that were added in MPI-2.0 and deprecated in MPI-2.2 */
typedef MPI_Comm_errhandler_function MPI_Comm_errhandler_fn;
typedef MPI_File_errhandler_function MPI_File_errhandler_fn;
typedef MPI_Win_errhandler_function MPI_Win_errhandler_fn;

/* Built in (0x1 in 30-31), errhandler (0x5 in bits 26-29, allkind (0
   in 22-25), index in the low bits */
static const int MPI_ERRORS_ARE_FATAL= ((MPI_Errhandler)0x54000000);
static const int MPI_ERRORS_RETURN   = ((MPI_Errhandler)0x54000001);
/* MPIR_ERRORS_THROW_EXCEPTIONS is not part of the MPI standard, it is here to
   facilitate the c++ binding which has MPI::ERRORS_THROW_EXCEPTIONS. 
   Using the MPIR prefix preserved the MPI_ names for objects defined by
   the standard. */
static const int MPIR_ERRORS_THROW_EXCEPTIONS= ((MPI_Errhandler)0x54000002);

/* Make the C names for the dup function mixed case.
   This is required for systems that use all uppercase names for Fortran 
   externals.  */
/* MPI 1 names */
//#define MPI_NULL_COPY_FN   ((MPI_Copy_function *)0)
//#define MPI_NULL_DELETE_FN ((MPI_Delete_function *)0)
//#define MPI_DUP_FN         MPIR_Dup_fn
/* MPI 2 names */
//#define MPI_COMM_NULL_COPY_FN ((MPI_Comm_copy_attr_function*)0)
//#define MPI_COMM_NULL_DELETE_FN ((MPI_Comm_delete_attr_function*)0)
//#define MPI_COMM_DUP_FN  ((MPI_Comm_copy_attr_function *)MPI_DUP_FN)
//#define MPI_WIN_NULL_COPY_FN ((MPI_Win_copy_attr_function*)0)
//#define MPI_WIN_NULL_DELETE_FN ((MPI_Win_delete_attr_function*)0)
//#define MPI_WIN_DUP_FN   ((MPI_Win_copy_attr_function*)MPI_DUP_FN)
//#define MPI_TYPE_NULL_COPY_FN ((MPI_Type_copy_attr_function*)0)
//#define MPI_TYPE_NULL_DELETE_FN ((MPI_Type_delete_attr_function*)0)
//#define MPI_TYPE_DUP_FN ((MPI_Type_copy_attr_function*)MPI_DUP_FN)

/* MPI request opjects */

/* MPI message objects for Mprobe and related functions */

/* User combination function */
typedef void (MPI_User_function) ( void *, void *, int *, MPI_Datatype * );

/* MPI Attribute copy and delete functions */
typedef int (MPI_Copy_function) ( MPI_Comm, int, void *, void *, void *, int * );
typedef int (MPI_Delete_function) ( MPI_Comm, int, void *, void * );

static const int MPI_VERSION   = 3;
static const int MPI_SUBVERSION= 0;
static const int MPICH_NAME    = 3;
static const int MPICH        = 1;
static const int MPICH_HAS_C2F = 1;


/* MPICH_VERSION is the version string. MPICH_NUMVERSION is the
 * numeric version that can be used in numeric comparisons.
 *
 * MPICH_VERSION uses the following format:
 * Version: [MAJ].[MIN].[REV][EXT][EXT_NUMBER]
 * Example: 1.0.7rc1 has
 *          MAJ = 1
 *          MIN = 0
 *          REV = 7
 *          EXT = rc
 *          EXT_NUMBER = 1
 *
 * MPICH_NUMVERSION will convert EXT to a format number:
 *          ALPHA (a) = 0
 *          BETA (b)  = 1
 *          RC (rc)   = 2
 *          PATCH (p) = 3
 * Regular releases are treated as patch 0
 *
 * Numeric version will have 1 digit for MAJ, 2 digits for MIN, 2
 * digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER. So,
 * 1.0.7rc1 will have the numeric version 10007201.
 */
/*
#define MPICH_VERSION "3.1.4"
#define MPICH_NUMVERSION 30104300

#define MPICH_RELEASE_TYPE_ALPHA  0
#define MPICH_RELEASE_TYPE_BETA   1
#define MPICH_RELEASE_TYPE_RC     2
#define MPICH_RELEASE_TYPE_PATCH  3

#define MPICH_CALC_VERSION(MAJOR, MINOR, REVISION, TYPE, PATCH) \
    (((MAJOR) * 10000000) + ((MINOR) * 100000) + ((REVISION) * 1000) + ((TYPE) * 100) + (PATCH))

#define MVAPICH2_VERSION "2.2rc1"
#define MVAPICH2_NUMVERSION 20200201

#define MVAPICH2_RELEASE_TYPE_ALPHA  0
#define MVAPICH2_RELEASE_TYPE_BETA   1
#define MVAPICH2_RELEASE_TYPE_RC     2
#define MVAPICH2_RELEASE_TYPE_PATCH  3

#define MVAPICH2_CALC_VERSION(MAJOR, MINOR, REVISION, TYPE, PATCH) \
    (((MAJOR) * 10000000) + ((MINOR) * 100000) + ((REVISION) * 1000) + ((TYPE) * 100) + (PATCH))
*/
/* for the datatype decoders */
enum MPIR_Combiner_enum {
    MPI_COMBINER_NAMED            = 1,
    MPI_COMBINER_DUP              = 2,
    MPI_COMBINER_CONTIGUOUS       = 3,
    MPI_COMBINER_VECTOR           = 4,
    MPI_COMBINER_HVECTOR_INTEGER  = 5,
    MPI_COMBINER_HVECTOR          = 6,
    MPI_COMBINER_INDEXED          = 7,
    MPI_COMBINER_HINDEXED_INTEGER = 8,
    MPI_COMBINER_HINDEXED         = 9,
    MPI_COMBINER_INDEXED_BLOCK    = 10,
    MPI_COMBINER_STRUCT_INTEGER   = 11,
    MPI_COMBINER_STRUCT           = 12,
    MPI_COMBINER_SUBARRAY         = 13,
    MPI_COMBINER_DARRAY           = 14,
    MPI_COMBINER_F90_REAL         = 15,
    MPI_COMBINER_F90_COMPLEX      = 16,
    MPI_COMBINER_F90_INTEGER      = 17,
    MPI_COMBINER_RESIZED          = 18,
    MPI_COMBINER_HINDEXED_BLOCK   = 19
};

/* for info */
typedef int MPI_Info;
static const int MPI_INFO_NULL        = ((MPI_Info)0x1c000000);
static const int MPI_INFO_ENV         = ((MPI_Info)0x5c000001);
static const int MPI_MAX_INFO_KEY      = 255;
static const int MPI_MAX_INFO_VAL     = 1024;

/* for subarray and darray constructors */
static const int MPI_ORDER_C             = 56;
static const int MPI_ORDER_FORTRAN       = 57;
static const int MPI_DISTRIBUTE_BLOCK   = 121;
static const int MPI_DISTRIBUTE_CYCLIC  = 122;
static const int MPI_DISTRIBUTE_NONE    = 123;
static const int MPI_DISTRIBUTE_DFLT_DARG= -49767;

static const int MPI_IN_PLACE = -1;

/* asserts for one-sided communication */
static const int MPI_MODE_NOCHECK     = 1024;
static const int MPI_MODE_NOSTORE     = 2048;
static const int MPI_MODE_NOPUT       = 4096;
static const int MPI_MODE_NOPRECEDE   = 8192;
static const int MPI_MODE_NOSUCCEED  = 16384;

/* predefined types for MPI_Comm_split_type */
//#define MPI_COMM_TYPE_SHARED    1

/* Definitions that are determined by configure. */
typedef long MPI_Aint;
typedef int MPI_Fint;
typedef long long MPI_Count;

/* Let ROMIO know that MPI_Offset is already defined */
//#define HAVE_MPI_OFFSET
/* MPI_OFFSET_TYPEDEF is set in configure and is 
      typedef $MPI_OFFSET MPI_Offset;
   where $MPI_OFFSET is the correct C type */
typedef long long MPI_Offset;

/* The order of these elements must match that in mpif.h, mpi_f08_types.f90,
   and mpi_c_interface_types.f90 */
typedef struct MPI_Status {
    int count_lo;
    int count_hi_and_cancelled;
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
} MPI_Status;

/* types for the MPI_T_ interface */
struct MPIR_T_enum_s;
struct MPIR_T_cvar_handle_s;
struct MPIR_T_pvar_handle_s;
struct MPIR_T_pvar_session_s;

typedef struct MPIR_T_enum_s * MPI_T_enum;
typedef struct MPIR_T_cvar_handle_s * MPI_T_cvar_handle;
typedef struct MPIR_T_pvar_handle_s * MPI_T_pvar_handle;
typedef struct MPIR_T_pvar_session_s * MPI_T_pvar_session;

/* extra const at front would be safer, but is incompatible with MPI_T_ prototypes */
extern struct MPIR_T_pvar_handle_s * const MPI_T_PVAR_ALL_HANDLES;

//static const MPI_T_enum MPI_T_ENUM_NULL        = ((MPI_T_enum)NULL);
//static const MPI_T_cvar_handle MPI_T_CVAR_HANDLE_NULL = ((MPI_T_cvar_handle)NULL);
//static const MPI_T_pvar_handle MPI_T_PVAR_HANDLE_NULL = ((MPI_T_pvar_handle)NULL);
//static const MPI_T_pvar_session MPI_T_PVAR_SESSION_NULL= ((MPI_T_pvar_session)NULL);

/* the MPI_T_ interface requires that these VERBOSITY constants occur in this
 * relative order with increasing values */
typedef enum MPIR_T_verbosity_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_VERBOSITY_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPI_T_VERBOSITY_USER_BASIC = 221,
    MPI_T_VERBOSITY_USER_DETAIL,
    MPI_T_VERBOSITY_USER_ALL,

    MPI_T_VERBOSITY_TUNER_BASIC,
    MPI_T_VERBOSITY_TUNER_DETAIL,
    MPI_T_VERBOSITY_TUNER_ALL,

    MPI_T_VERBOSITY_MPIDEV_BASIC,
    MPI_T_VERBOSITY_MPIDEV_DETAIL,
    MPI_T_VERBOSITY_MPIDEV_ALL
} MPIR_T_verbosity_t;

typedef enum MPIR_T_bind_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_BIND_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPI_T_BIND_NO_OBJECT = 9700,
    MPI_T_BIND_MPI_COMM,
    MPI_T_BIND_MPI_DATATYPE,
    MPI_T_BIND_MPI_ERRHANDLER,
    MPI_T_BIND_MPI_FILE,
    MPI_T_BIND_MPI_GROUP,
    MPI_T_BIND_MPI_OP,
    MPI_T_BIND_MPI_REQUEST,
    MPI_T_BIND_MPI_WIN,
    MPI_T_BIND_MPI_MESSAGE,
    MPI_T_BIND_MPI_INFO
} MPIR_T_bind_t;

typedef enum MPIR_T_scope_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_SCOPE_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPI_T_SCOPE_CONSTANT = 60438,
    MPI_T_SCOPE_READONLY,
    MPI_T_SCOPE_LOCAL,
    MPI_T_SCOPE_GROUP,
    MPI_T_SCOPE_GROUP_EQ,
    MPI_T_SCOPE_ALL,
    MPI_T_SCOPE_ALL_EQ
} MPIR_T_scope_t;

typedef enum MPIR_T_pvar_class_t {
    /* don't name-shift this if/when MPI_T_ is accepted, this is an MPICH-only
     * extension */
    MPIX_T_PVAR_CLASS_INVALID = 0,

    /* arbitrarily shift values to aid debugging and reduce accidental errors */
    MPIR_T_PVAR_CLASS_FIRST = 240,
    MPI_T_PVAR_CLASS_STATE = MPIR_T_PVAR_CLASS_FIRST,
    MPI_T_PVAR_CLASS_LEVEL,
    MPI_T_PVAR_CLASS_SIZE,
    MPI_T_PVAR_CLASS_PERCENTAGE,
    MPI_T_PVAR_CLASS_HIGHWATERMARK,
    MPI_T_PVAR_CLASS_LOWWATERMARK,
    MPI_T_PVAR_CLASS_COUNTER,
    MPI_T_PVAR_CLASS_AGGREGATE,
    MPI_T_PVAR_CLASS_TIMER,
    MPI_T_PVAR_CLASS_GENERIC,
    MPIR_T_PVAR_CLASS_LAST,
    MPIR_T_PVAR_CLASS_NUMBER = MPIR_T_PVAR_CLASS_LAST - MPIR_T_PVAR_CLASS_FIRST
} MPIR_T_pvar_class_t;

/* Handle conversion types/functions */

/* Programs that need to convert types used in MPICH should use these */
//static const int MPI_Comm_c2f(comm)= (MPI_Fint)(comm);
//static const int MPI_Comm_f2c(comm)= (MPI_Comm)(comm);
//static const int MPI_Type_c2f(datatype)= (MPI_Fint)(datatype);
//static const int MPI_Type_f2c(datatype)= (MPI_Datatype)(datatype);
//static const int MPI_Group_c2f(group)= (MPI_Fint)(group);
//static const int MPI_Group_f2c(group)= (MPI_Group)(group);
//static const int MPI_Info_c2f(info)= (MPI_Fint)(info);
//static const int MPI_Info_f2c(info)= (MPI_Info)(info);
//static const int MPI_Request_f2c(request)= (MPI_Request)(request);
//static const int MPI_Request_c2f(request)= (MPI_Fint)(request);
//static const int MPI_Op_c2f(op)= (MPI_Fint)(op);
//static const int MPI_Op_f2c(op)= (MPI_Op)(op);
//static const int MPI_Errhandler_c2f(errhandler)= (MPI_Fint)(errhandler);
//static const int MPI_Errhandler_f2c(errhandler)= (MPI_Errhandler)(errhandler);
//static const int MPI_Win_c2f(win)  = (MPI_Fint)(win);
//static const int MPI_Win_f2c(win)  = (MPI_Win)(win);
//static const int MPI_Message_c2f(msg)= ((MPI_Fint)(msg));
//static const int MPI_Message_f2c(msg)= ((MPI_Message)(msg));

static const int MPI_STATUS_IGNORE= 1;
static const int MPI_STATUSES_IGNORE= 1;
static const int MPI_ERRCODES_IGNORE= 0;

/* See 4.12.5 for MPI_F_STATUS(ES)_IGNORE */
// MPIU_DLL_SPEC
//extern MPIU_DLL_SPEC MPI_Fint * MPI_F_STATUS_IGNORE;
//extern MPIU_DLL_SPEC MPI_Fint * MPI_F_STATUSES_IGNORE;
/* The annotation MPIU_DLL_SPEC to the extern statements is used 
   as a hook for systems that require C extensions to correctly construct
   DLLs, and is defined as an empty string otherwise
 */

/* The MPI standard requires that the ARGV_NULL values be the same as
   NULL (see 5.3.2) */
static const int MPI_ARGV_NULL= 0;
static const int MPI_ARGVS_NULL= 0;

/* For supported thread levels */
static const int MPI_THREAD_SINGLE= 0;
static const int MPI_THREAD_FUNNELED= 1;
static const int MPI_THREAD_SERIALIZED= 2;
static const int MPI_THREAD_MULTIPLE= 3;

/* Typedefs for generalized requests */
typedef int (MPI_Grequest_cancel_function)(void *, int);
typedef int (MPI_Grequest_free_function)(void *);
typedef int (MPI_Grequest_query_function)(void *, MPI_Status *);
typedef int (MPIX_Grequest_poll_function)(void *, MPI_Status *);
typedef int (MPIX_Grequest_wait_function)(int, void **, double, MPI_Status *);

/* MPI's error classes */
static const int MPI_SUCCESS         = 0;      /* Successful return code */
/* Communication argument parameters */
static const int MPI_ERR_BUFFER      = 1;      /* Invalid buffer pointer */
static const int MPI_ERR_COUNT       = 2;      /* Invalid count argument */
static const int MPI_ERR_TYPE        = 3;      /* Invalid datatype argument */
static const int MPI_ERR_TAG         = 4;      /* Invalid tag argument */
static const int MPI_ERR_COMM        = 5;      /* Invalid communicator */
static const int MPI_ERR_RANK        = 6;      /* Invalid rank */
static const int MPI_ERR_ROOT        = 7;      /* Invalid root */
static const int MPI_ERR_TRUNCATE   = 14;      /* Message truncated on receive */

/* MPI Objects (other than COMM) */
static const int MPI_ERR_GROUP       = 8;      /* Invalid group */
static const int MPI_ERR_OP          = 9;      /* Invalid operation */
static const int MPI_ERR_REQUEST    = 19;      /* Invalid mpi_request handle */

/* Special topology argument parameters */
static const int MPI_ERR_TOPOLOGY   = 10;      /* Invalid topology */
static const int MPI_ERR_DIMS       = 11;      /* Invalid dimension argument */

/* All other arguments.  This is a class with many kinds */
static const int MPI_ERR_ARG        = 12;      /* Invalid argument */

/* Other errors that are not simply an invalid argument */
static const int MPI_ERR_OTHER      = 15;      /* Other error; use Error_string */

static const int MPI_ERR_UNKNOWN    = 13;      /* Unknown error */
static const int MPI_ERR_INTERN     = 16;      /* Internal error code    */

/* Multiple completion has three special error classes */
static const int MPI_ERR_IN_STATUS          = 17;      /* Look in status for error value */
static const int MPI_ERR_PENDING            = 18;      /* Pending request */

/* New MPI-2 Error classes */
static const int MPI_ERR_ACCESS     = 20;      /* */
static const int MPI_ERR_AMODE      = 21;      /* */
static const int MPI_ERR_BAD_FILE   = 22;      /* */
static const int MPI_ERR_CONVERSION = 23;      /* */
static const int MPI_ERR_DUP_DATAREP= 24;      /* */
static const int MPI_ERR_FILE_EXISTS= 25;      /* */
static const int MPI_ERR_FILE_IN_USE= 26;      /* */
static const int MPI_ERR_FILE       = 27;      /* */
static const int MPI_ERR_IO         = 32;      /* */
static const int MPI_ERR_NO_SPACE   = 36;      /* */
static const int MPI_ERR_NO_SUCH_FILE= 37;     /* */
static const int MPI_ERR_READ_ONLY  = 40;      /* */
static const int MPI_ERR_UNSUPPORTED_DATAREP  = 43;  /* */

/* MPI_ERR_INFO is NOT defined in the MPI-2 standard.  I believe that
   this is an oversight */
static const int MPI_ERR_INFO       = 28;      /* */
static const int MPI_ERR_INFO_KEY   = 29;      /* */
static const int MPI_ERR_INFO_VALUE = 30;      /* */
static const int MPI_ERR_INFO_NOKEY = 31;      /* */

static const int MPI_ERR_NAME       = 33;      /* */
static const int MPI_ERR_NO_MEM     = 34;      /* Alloc_mem could not allocate memory */
static const int MPI_ERR_NOT_SAME   = 35;      /* */
static const int MPI_ERR_PORT       = 38;      /* */
static const int MPI_ERR_QUOTA      = 39;      /* */
static const int MPI_ERR_SERVICE    = 41;      /* */
static const int MPI_ERR_SPAWN      = 42;      /* */
static const int MPI_ERR_UNSUPPORTED_OPERATION= 44; /* */
static const int MPI_ERR_WIN        = 45;      /* */

static const int MPI_ERR_BASE       = 46;      /* */
static const int MPI_ERR_LOCKTYPE   = 47;      /* */
static const int MPI_ERR_KEYVAL     = 48;      /* Erroneous attribute key */
static const int MPI_ERR_RMA_CONFLICT= 49;     /* */
static const int MPI_ERR_RMA_SYNC   = 50;      /* */ 
static const int MPI_ERR_SIZE       = 51;      /* */
static const int MPI_ERR_DISP       = 52;      /* */
static const int MPI_ERR_ASSERT     = 53;      /* */

static const int MPI_ERR_RMA_RANGE = 55;       /* */
static const int MPI_ERR_RMA_ATTACH= 56;       /* */
static const int MPI_ERR_RMA_SHARED= 57;       /* */
static const int MPI_ERR_RMA_FLAVOR= 58;       /* */

/* Return codes for functions in the MPI Tool Information Interface */
static const int MPI_T_ERR_MEMORY           = 59;  /* Out of memory */
static const int MPI_T_ERR_NOT_INITIALIZED  = 60;  /* Interface not initialized */
static const int MPI_T_ERR_CANNOT_INIT      = 61;  /* Interface not in the state to
                                           be initialized */
static const int MPI_T_ERR_INVALID_INDEX    = 62;  /* The index is invalid or
                                           has been deleted  */
static const int MPI_T_ERR_INVALID_ITEM     = 63;  /* Item index queried is out of range */
static const int MPI_T_ERR_INVALID_HANDLE   = 64;  /* The handle is invalid */
static const int MPI_T_ERR_OUT_OF_HANDLES   = 65;  /* No more handles available */
static const int MPI_T_ERR_OUT_OF_SESSIONS  = 66;  /* No more sessions available */
static const int MPI_T_ERR_INVALID_SESSION  = 67;  /* Session argument is not valid */
static const int MPI_T_ERR_CVAR_SET_NOT_NOW = 68;  /* Cvar can't be set at this moment */
static const int MPI_T_ERR_CVAR_SET_NEVER   = 69;  /* Cvar can't be set until
                                           end of execution */
static const int MPI_T_ERR_PVAR_NO_STARTSTOP= 70;  /* Pvar can't be started or stopped */
static const int MPI_T_ERR_PVAR_NO_WRITE    = 71;  /* Pvar can't be written or reset */
static const int MPI_T_ERR_PVAR_NO_ATOMIC   = 72;  /* Pvar can't be R/W atomically */


static const int MPI_ERR_LASTCODE   = 0x3fffffff;  /* Last valid error code for a 
                                           predefined error class */
/* WARNING: this is also defined in mpishared.h.  Update both locations */
static const int MPICH_ERR_LAST_CLASS= 72;     /* It is also helpful to know the
                                       last valid class */

static const int MPICH_ERR_FIRST_MPIX= 100; /* Define a gap here because sock is
                                  * already using some of the values in this
                                  * range. All MPIX error codes will be
                                  * above this value to be ABI complaint. */

static const int MPIX_ERR_PROC_FAILED         = MPICH_ERR_FIRST_MPIX+1; /* Process failure */
static const int MPIX_ERR_PROC_FAILED_PENDING = MPICH_ERR_FIRST_MPIX+2; /* A failure has caused this request
                                                              * to be pending */
static const int MPIX_ERR_REVOKED             = MPICH_ERR_FIRST_MPIX+3; /* The communciation object has been revoked */

static const int MPICH_ERR_LAST_MPIX          = MPICH_ERR_FIRST_MPIX+3;


/* End of MPI's error classes */

/* Function type defs */
typedef int (MPI_Datarep_conversion_function)(void *, MPI_Datatype, int,
             void *, MPI_Offset, void *);
typedef int (MPI_Datarep_extent_function)(MPI_Datatype datatype, MPI_Aint *,
                      void *);
static const int MPI_CONVERSION_FN_NULL= 0;

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Buffer_attach(void *buffer, int size);
int MPI_Buffer_detach(void *buffer_addr, int *size);
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
               MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Wait(MPI_Request *request, MPI_Status *status);
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int MPI_Request_free(MPI_Request *request);
int MPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status *status);
int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx, int *flag,
                MPI_Status *status);
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag,
                MPI_Status array_of_statuses[]);
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount,
                 int array_of_indices[], MPI_Status array_of_statuses[]);
int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount,
                 int array_of_indices[], MPI_Status array_of_statuses[]);
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Cancel(MPI_Request *request);
int MPI_Test_cancelled(const MPI_Status *status, int *flag);
int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                  MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                  MPI_Comm comm, MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Start(MPI_Request *request);
int MPI_Startall(int count, MPI_Request array_of_requests[]);
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
                 int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag, MPI_Comm comm, MPI_Status *status)
                 __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,6,8)));
int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest,
                         int sendtag, int source, int recvtag, MPI_Comm comm,
                         MPI_Status *status) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                    MPI_Datatype *newtype);
int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype,
                     MPI_Datatype *newtype);
int MPI_Type_indexed(int count, const int *array_of_blocklengths,
                     const int *array_of_displacements, MPI_Datatype oldtype,
                     MPI_Datatype *newtype);
int MPI_Type_hindexed(int count, const int *array_of_blocklengths,
                      const MPI_Aint *array_of_displacements, MPI_Datatype oldtype,
                      MPI_Datatype *newtype);
int MPI_Type_struct(int count, const int *array_of_blocklengths,
                    const MPI_Aint *array_of_displacements,
                    const MPI_Datatype *array_of_types, MPI_Datatype *newtype);
int MPI_Address(const void *location, MPI_Aint *address);
int MPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
int MPI_Type_size(MPI_Datatype datatype, int *size);
int MPI_Type_lb(MPI_Datatype datatype, MPI_Aint *displacement);
int MPI_Type_ub(MPI_Datatype datatype, MPI_Aint *displacement);
int MPI_Type_commit(MPI_Datatype *datatype);
int MPI_Type_free(MPI_Datatype *datatype);
int MPI_Get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count);
int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf,
             int outsize, int *position, MPI_Comm comm) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Unpack(const void *inbuf, int insize, int *position, void *outbuf, int outcount,
               MPI_Datatype datatype, MPI_Comm comm) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size);
int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
              __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
               int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
               __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                const int *recvcounts, const int *displs, MPI_Datatype recvtype, int root,
                MPI_Comm comm)
                __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,7)));
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
                __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
                 __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,5,7)));
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                  __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   const int *recvcounts, const int *displs, MPI_Datatype recvtype, MPI_Comm comm)
                   __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,7)));
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                 __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Alltoallv(const void *sendbuf, const int *sendcounts, const int *sdispls,
                  MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
                  const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
                  __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,5,8)));
int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[],
                  const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                  const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm);
int MPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, MPI_Comm comm)
               __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm)
               __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op);
int MPI_Op_free(MPI_Op *op);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm)
                  __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
                       __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
             MPI_Comm comm)
             __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Group_size(MPI_Group group, int *size);
int MPI_Group_rank(MPI_Group group, int *rank);
int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2,
                              int ranks2[]);
int MPI_Group_compare(MPI_Group group1, MPI_Group group2, int *result);
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
int MPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
int MPI_Group_free(MPI_Group *group);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result);
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm);
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
int MPI_Comm_free(MPI_Comm *comm);
int MPI_Comm_test_inter(MPI_Comm comm, int *flag);
int MPI_Comm_remote_size(MPI_Comm comm, int *size);
int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm,
                         int remote_leader, int tag, MPI_Comm *newintercomm);
int MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm);
int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                      int *keyval, void *extra_state);
int MPI_Keyval_free(int *keyval);
int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val);
int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
int MPI_Attr_delete(MPI_Comm comm, int keyval);
int MPI_Topo_test(MPI_Comm comm, int *status);
int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[],
                    int reorder, MPI_Comm *comm_cart);
int MPI_Dims_create(int nnodes, int ndims, int dims[]);
int MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int indx[], const int edges[],
                     int reorder, MPI_Comm *comm_graph);
int MPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges);
int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int indx[], int edges[]);
int MPI_Cartdim_get(MPI_Comm comm, int *ndims);
int MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int MPI_Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[]);
int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm);
int MPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int *newrank);
int MPI_Graph_map(MPI_Comm comm, int nnodes, const int indx[], const int edges[], int *newrank);
int MPI_Get_processor_name(char *name, int *resultlen);
int MPI_Get_version(int *version, int *subversion);
int MPI_Get_library_version(char *version, int *resultlen);
int MPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler);
int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
int MPI_Errhandler_free(MPI_Errhandler *errhandler);
int MPI_Error_string(int errorcode, char *string, int *resultlen);
int MPI_Error_class(int errorcode, int *errorclass);
double MPI_Wtime(void);
double MPI_Wtick(void);
int MPI_Init(int *argc, char ***argv);
int MPI_Finalize(void);
int MPI_Initialized(int *flag);
int MPI_Abort(MPI_Comm comm, int errorcode);



int MPI_Pcontrol(const int level, ...);
int MPI_DUP_FN(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in,
               void *attribute_val_out, int *flag);


int MPI_Close_port(const char *port_name);
int MPI_Comm_accept(const char *port_name, MPI_Info info, int root, MPI_Comm comm,
                    MPI_Comm *newcomm);
int MPI_Comm_connect(const char *port_name, MPI_Info info, int root, MPI_Comm comm,
                     MPI_Comm *newcomm);
int MPI_Comm_disconnect(MPI_Comm *comm);
int MPI_Comm_get_parent(MPI_Comm *parent);
int MPI_Comm_join(int fd, MPI_Comm *intercomm);
int MPI_Comm_spawn(const char *command, char *argv[], int maxprocs, MPI_Info info, int root,
                   MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
int MPI_Comm_spawn_multiple(int count, char *array_of_commands[], char **array_of_argv[],
                            const int array_of_maxprocs[], const MPI_Info array_of_info[],
                            int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
int MPI_Lookup_name(const char *service_name, MPI_Info info, char *port_name);
int MPI_Open_port(MPI_Info info, char *port_name);
int MPI_Publish_name(const char *service_name, MPI_Info info, const char *port_name);
int MPI_Unpublish_name(const char *service_name, MPI_Info info, const char *port_name);
int MPI_Comm_set_info(MPI_Comm comm, MPI_Info info);
int MPI_Comm_get_info(MPI_Comm comm, MPI_Info *info);


int MPI_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                   int target_rank, MPI_Aint target_disp, int target_count,
                   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
                   __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Win_complete(MPI_Win win);
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
                   MPI_Win *win);
int MPI_Win_fence(int assert, MPI_Win win);
int MPI_Win_free(MPI_Win *win);
int MPI_Win_get_group(MPI_Win win, MPI_Group *group);
int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);
int MPI_Win_post(MPI_Group group, int assert, MPI_Win win);
int MPI_Win_start(MPI_Group group, int assert, MPI_Win win);
int MPI_Win_test(MPI_Win win, int *flag);
int MPI_Win_unlock(int rank, MPI_Win win);
int MPI_Win_wait(MPI_Win win);


int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr,
                     MPI_Win *win);
int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
                            void *baseptr, MPI_Win *win);
int MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr);
int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win);
int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size);
int MPI_Win_detach(MPI_Win win, const void *base);
int MPI_Win_get_info(MPI_Win win, MPI_Info *info_used);
int MPI_Win_set_info(MPI_Win win, MPI_Info info);
int MPI_Get_accumulate(const void *origin_addr, int origin_count,
                        MPI_Datatype origin_datatype, void *result_addr, int result_count,
                        MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                        int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
                        __attribute__((pointer_with_type_tag(MPI,1,3)))
                        __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                      MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
                      MPI_Op op, MPI_Win win)
                      __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                          void *result_addr, MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPI_Win win)
                          __attribute__((pointer_with_type_tag(MPI,1,4)))
                          __attribute__((pointer_with_type_tag(MPI,2,4)))
                          __attribute__((pointer_with_type_tag(MPI,3,4)));
int MPI_Rput(const void *origin_addr, int origin_count,
              MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype, MPI_Win win,
              MPI_Request *request)
              __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Rget(void *origin_addr, int origin_count,
              MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype, MPI_Win win,
              MPI_Request *request)
              __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Raccumulate(const void *origin_addr, int origin_count,
                     MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
                     int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
                     MPI_Request *request)
                     __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Rget_accumulate(const void *origin_addr, int origin_count,
                         MPI_Datatype origin_datatype, void *result_addr, int result_count,
                         MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                         int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win,
                         MPI_Request *request)
                         __attribute__((pointer_with_type_tag(MPI,1,3)))
                         __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Win_lock_all(int assert, MPI_Win win);
int MPI_Win_unlock_all(MPI_Win win);
int MPI_Win_flush(int rank, MPI_Win win);
int MPI_Win_flush_all(MPI_Win win);
int MPI_Win_flush_local(int rank, MPI_Win win);
int MPI_Win_flush_local_all(MPI_Win win);
int MPI_Win_sync(MPI_Win win);


int MPI_Add_error_class(int *errorclass);
int MPI_Add_error_code(int errorclass, int *errorcode);
int MPI_Add_error_string(int errorcode, const char *string);
int MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode);
int MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *comm_copy_attr_fn,
                           MPI_Comm_delete_attr_function *comm_delete_attr_fn, int *comm_keyval,
                           void *extra_state);
int MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval);
int MPI_Comm_free_keyval(int *comm_keyval);
int MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag);
int MPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen);
int MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val);
int MPI_Comm_set_name(MPI_Comm comm, const char *comm_name);
int MPI_File_call_errhandler(MPI_File fh, int errorcode);
int MPI_Grequest_complete(MPI_Request request);
int MPI_Grequest_start(MPI_Grequest_query_function *query_fn, MPI_Grequest_free_function *free_fn,
                       MPI_Grequest_cancel_function *cancel_fn, void *extra_state,
                       MPI_Request *request);
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Is_thread_main(int *flag);
int MPI_Query_thread(int *provided);
int MPI_Status_set_cancelled(MPI_Status *status, int flag);
int MPI_Status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count);
int MPI_Type_create_keyval(MPI_Type_copy_attr_function *type_copy_attr_fn,
                           MPI_Type_delete_attr_function *type_delete_attr_fn,
                           int *type_keyval, void *extra_state);
int MPI_Type_delete_attr(MPI_Datatype datatype, int type_keyval);
int MPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_free_keyval(int *type_keyval);
int MPI_Type_get_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val, int *flag);
int MPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses,
                          int max_datatypes, int array_of_integers[],
                          MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[]);
int MPI_Type_get_envelope(MPI_Datatype datatype, int *num_integers, int *num_addresses,
                          int *num_datatypes, int *combiner);
int MPI_Type_get_name(MPI_Datatype datatype, char *type_name, int *resultlen);
int MPI_Type_set_attr(MPI_Datatype datatype, int type_keyval, void *attribute_val);
int MPI_Type_set_name(MPI_Datatype datatype, const char *type_name);
int MPI_Type_match_size(int typeclass, int size, MPI_Datatype *datatype);
int MPI_Win_call_errhandler(MPI_Win win, int errorcode);
int MPI_Win_create_keyval(MPI_Win_copy_attr_function *win_copy_attr_fn,
                          MPI_Win_delete_attr_function *win_delete_attr_fn, int *win_keyval,
                          void *extra_state);
int MPI_Win_delete_attr(MPI_Win win, int win_keyval);
int MPI_Win_free_keyval(int *win_keyval);
int MPI_Win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag);
int MPI_Win_get_name(MPI_Win win, char *win_name, int *resultlen);
int MPI_Win_set_attr(MPI_Win win, int win_keyval, void *attribute_val);
int MPI_Win_set_name(MPI_Win win, const char *win_name);

int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr);
int MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *comm_errhandler_fn,
                               MPI_Errhandler *errhandler);
int MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler);
int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);
int MPI_File_create_errhandler(MPI_File_errhandler_function *file_errhandler_fn,
                               MPI_Errhandler *errhandler);
int MPI_File_get_errhandler(MPI_File file, MPI_Errhandler *errhandler);
int MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler);
int MPI_Finalized(int *flag);
int MPI_Free_mem(void *base);
int MPI_Get_address(const void *location, MPI_Aint *address);
int MPI_Info_create(MPI_Info *info);
int MPI_Info_delete(MPI_Info info, const char *key);
int MPI_Info_dup(MPI_Info info, MPI_Info *newinfo);
int MPI_Info_free(MPI_Info *info);
int MPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag);
int MPI_Info_get_nkeys(MPI_Info info, int *nkeys);
int MPI_Info_get_nthkey(MPI_Info info, int n, char *key);
int MPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag);
int MPI_Info_set(MPI_Info info, const char *key, const char *value);
int MPI_Pack_external(const char datarep[], const void *inbuf, int incount,
                      MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position)
                      __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Pack_external_size(const char datarep[], int incount, MPI_Datatype datatype,
                           MPI_Aint *size);
int MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status);
int MPI_Status_c2f(const MPI_Status *c_status, MPI_Fint *f_status);
int MPI_Status_f2c(const MPI_Fint *f_status, MPI_Status *c_status);
int MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[],
                           const int array_of_distribs[], const int array_of_dargs[],
                           const int array_of_psizes[], int order, MPI_Datatype oldtype,
                           MPI_Datatype *newtype);
int MPI_Type_create_hindexed(int count, const int array_of_blocklengths[],
                             const MPI_Aint array_of_displacements[], MPI_Datatype oldtype,
                             MPI_Datatype *newtype);
int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype,
                            MPI_Datatype *newtype);
int MPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[],
                                  MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_create_hindexed_block(int count, int blocklength,
                                   const MPI_Aint array_of_displacements[],
                                   MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
                            MPI_Datatype *newtype);
int MPI_Type_create_struct(int count, const int array_of_blocklengths[],
                           const MPI_Aint array_of_displacements[],
                           const MPI_Datatype array_of_types[], MPI_Datatype *newtype);
int MPI_Type_create_subarray(int ndims, const int array_of_sizes[],
                             const int array_of_subsizes[], const int array_of_starts[],
                             int order, MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent);
int MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent);
int MPI_Unpack_external(const char datarep[], const void *inbuf, MPI_Aint insize,
                        MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype)
                        __attribute__((pointer_with_type_tag(MPI,5,7)));
int MPI_Win_create_errhandler(MPI_Win_errhandler_function *win_errhandler_fn,
                              MPI_Errhandler *errhandler);
int MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler);
int MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler);


int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message,
                MPI_Status *status);
int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message,
               MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status);
int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message,
              MPI_Status *status) __attribute__((pointer_with_type_tag(MPI,1,3)));


int MPI_Comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request);
int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request);
int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm,
               MPI_Request *request) __attribute__((pointer_with_type_tag(MPI,1,3)));
int MPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
                MPI_Request *request)
                __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root,
                 MPI_Comm comm, MPI_Request *request)
                 __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,7)));
int MPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm,
                 MPI_Request *request)
                 __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Iscatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm, MPI_Request *request)
                  __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,5,7)));
int MPI_Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                   __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                    const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                    MPI_Comm comm, MPI_Request *request)
                    __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,7)));
int MPI_Ialltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                  __attribute__((pointer_with_type_tag(MPI,1,3))) __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Ialltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                   MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                   const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                   MPI_Request *request)
                   __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,5,8)));
int MPI_Ialltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[],
                   const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                   const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm,
                   MPI_Request *request);
int MPI_Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, int root, MPI_Comm comm, MPI_Request *request)
                __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                   MPI_Op op, MPI_Comm comm, MPI_Request *request)
                   __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Ireduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request)
                        __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount,
                              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                              MPI_Request *request)
                              __attribute__((pointer_with_type_tag(MPI,1,4)))
                              __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
              MPI_Comm comm, MPI_Request *request)
              __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));
int MPI_Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm comm, MPI_Request *request)
                __attribute__((pointer_with_type_tag(MPI,1,4))) __attribute__((pointer_with_type_tag(MPI,2,4)));


int MPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, int recvcount, MPI_Datatype recvtype,
                            MPI_Comm comm, MPI_Request *request)
                            __attribute__((pointer_with_type_tag(MPI,1,3)))
                            __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Ineighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, const int recvcounts[], const int displs[],
                             MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
                             __attribute__((pointer_with_type_tag(MPI,1,3)))
                             __attribute__((pointer_with_type_tag(MPI,4,7)));
int MPI_Ineighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm,
                           MPI_Request *request)
                           __attribute__((pointer_with_type_tag(MPI,1,3)))
                           __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                            MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                            MPI_Request *request)
                            __attribute__((pointer_with_type_tag(MPI,1,4)))
                            __attribute__((pointer_with_type_tag(MPI,5,8)));
int MPI_Ineighbor_alltoallw(const void *sendbuf, const int sendcounts[],
                            const MPI_Aint sdispls[], const MPI_Datatype sendtypes[],
                            void *recvbuf, const int recvcounts[], const MPI_Aint rdispls[],
                            const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request *request);
int MPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                           __attribute__((pointer_with_type_tag(MPI,1,3)))
                           __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Neighbor_allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, const int recvcounts[], const int displs[],
                            MPI_Datatype recvtype, MPI_Comm comm)
                            __attribute__((pointer_with_type_tag(MPI,1,3)))
                            __attribute__((pointer_with_type_tag(MPI,4,7)));
int MPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                          void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                          __attribute__((pointer_with_type_tag(MPI,1,3)))
                          __attribute__((pointer_with_type_tag(MPI,4,6)));
int MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                           MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                           const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
                           __attribute__((pointer_with_type_tag(MPI,1,4)))
                           __attribute__((pointer_with_type_tag(MPI,5,8)));
int MPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[],
                           const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                           const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm);


int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm);


int MPI_Get_elements_x(const MPI_Status *status, MPI_Datatype datatype, MPI_Count *count);
int MPI_Status_set_elements_x(MPI_Status *status, MPI_Datatype datatype, MPI_Count count);
int MPI_Type_get_extent_x(MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent);
int MPI_Type_get_true_extent_x(MPI_Datatype datatype, MPI_Count *lb, MPI_Count *extent);
int MPI_Type_size_x(MPI_Datatype datatype, MPI_Count *size);


int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm);





int MPI_T_init_thread(int required, int *provided);
int MPI_T_finalize(void);
int MPI_T_enum_get_info(MPI_T_enum enumtype, int *num, char *name, int *name_len);
int MPI_T_enum_get_item(MPI_T_enum enumtype, int indx, int *value, char *name, int *name_len);
int MPI_T_cvar_get_num(int *num_cvar);
int MPI_T_cvar_get_info(int cvar_index, char *name, int *name_len, int *verbosity,
                        MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                        int *binding, int *scope);
int MPI_T_cvar_handle_alloc(int cvar_index, void *obj_handle, MPI_T_cvar_handle *handle,
                            int *count);
int MPI_T_cvar_handle_free(MPI_T_cvar_handle *handle);
int MPI_T_cvar_read(MPI_T_cvar_handle handle, void *buf);
int MPI_T_cvar_write(MPI_T_cvar_handle handle, const void *buf);
int MPI_T_pvar_get_num(int *num_pvar);
int MPI_T_pvar_get_info(int pvar_index, char *name, int *name_len, int *verbosity, int *var_class,
                        MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len,
                        int *binding, int *readonly, int *continuous, int *atomic);
int MPI_T_pvar_session_create(MPI_T_pvar_session *session);
int MPI_T_pvar_session_free(MPI_T_pvar_session *session);
int MPI_T_pvar_handle_alloc(MPI_T_pvar_session session, int pvar_index, void *obj_handle,
                            MPI_T_pvar_handle *handle, int *count);
int MPI_T_pvar_handle_free(MPI_T_pvar_session session, MPI_T_pvar_handle *handle);
int MPI_T_pvar_start(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int MPI_T_pvar_stop(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int MPI_T_pvar_read(MPI_T_pvar_session session, MPI_T_pvar_handle handle, void *buf);
int MPI_T_pvar_write(MPI_T_pvar_session session, MPI_T_pvar_handle handle, const void *buf);
int MPI_T_pvar_reset(MPI_T_pvar_session session, MPI_T_pvar_handle handle);
int MPI_T_pvar_readreset(MPI_T_pvar_session session, MPI_T_pvar_handle handle, void *buf);
int MPI_T_category_get_num(int *num_cat);
int MPI_T_category_get_info(int cat_index, char *name, int *name_len, char *desc, int *desc_len,
                            int *num_cvars, int *num_pvars, int *num_categories);
int MPI_T_category_get_cvars(int cat_index, int len, int indices[]);
int MPI_T_category_get_pvars(int cat_index, int len, int indices[]);
int MPI_T_category_get_categories(int cat_index, int len, int indices[]);
int MPI_T_category_changed(int *stamp);
/* End Skip Prototypes */
/* Non-standard but public extensions to MPI */
/* Fault Tolerance Extensions */
int MPIX_Comm_failure_ack(MPI_Comm comm);
int MPIX_Comm_failure_get_acked(MPI_Comm comm, MPI_Group *failedgrp);
int MPIX_Comm_revoke(MPI_Comm comm);
int MPIX_Comm_shrink(MPI_Comm comm, MPI_Comm *newcomm);
int MPIX_Comm_agree(MPI_Comm comm, int *flag);

/* feature advertisement */
static const int MPIIMPL_ADVERTISES_FEATURES= 1;
static const int MPIIMPL_HAVE_MPI_INFO= 1;
static const int MPIIMPL_HAVE_MPI_COMBINER_DARRAY= 1;
static const int MPIIMPL_HAVE_MPI_TYPE_CREATE_DARRAY= 1;
static const int MPIIMPL_HAVE_MPI_COMBINER_SUBARRAY= 1;
static const int MPIIMPL_HAVE_MPI_COMBINER_DUP= 1;
static const int MPIIMPL_HAVE_MPI_GREQUEST= 1;
static const int MPIIMPL_HAVE_STATUS_SET_BYTES= 1;
static const int MPIIMPL_HAVE_STATUS_SET_INFO= 1;

]]

-- Dynamically link the MPI library into the global namespace if needed.
local C = ffi.C
if not pcall(function() return ffi.C.MPI_Init end) then
  C = ffi.load("libmpi.so", true)
end

local _M = {}

-- C types
local MPI_Aint_1     = ffi.typeof("MPI_Aint[1]")
local MPI_Comm_1     = ffi.typeof("MPI_Comm[1]")
local MPI_Datatype   = ffi.typeof("MPI_Datatype")
local MPI_Datatype_1 = ffi.typeof("MPI_Datatype[1]")
local MPI_Op         = ffi.typeof("MPI_Op")
local char_n         = ffi.typeof("char[?]")
local int_1          = ffi.typeof("int[1]")
local int_n          = ffi.typeof("int[?]")

-- MPI constants
local MPI_COMM_WORLD    = ffi.cast("MPI_Comm", C.MPI_COMM_WORLD)
local MPI_COMM_NULL     = ffi.cast("MPI_Comm", C.MPI_COMM_NULL)
local MPI_ERRORS_RETURN = ffi.cast("MPI_Errhandler", C.MPI_ERRORS_RETURN)
local MPI_STATUS_IGNORE = ffi.cast("MPI_Status *", C.MPI_STATUS_IGNORE)

-- Initialise MPI library.
assert(C.MPI_Init(nil, nil) == C.MPI_SUCCESS)
-- Gracefully abort program on error.
assert(C.MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == C.MPI_SUCCESS)

------------------------------------------------------------------------------
-- Environment.
------------------------------------------------------------------------------

local error_string do
  local buf = char_n(C.MPI_MAX_ERROR_STRING)
  local len = int_1()
  function error_string(err)
    assert(C.MPI_Error_string(err, buf, len) == C.MPI_SUCCESS)
    return ffi.string(buf, len[0])
  end
end

function _M.finalize()
  local err = C.MPI_Finalize()
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.finalized()
    local flag = int_1()
    local err = C.MPI_Finalized(flag)
    if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
    return flag[0] ~= 0
end

function _M.get_version()
    local version, subversion = int_1(), int_1()
    local err = C.MPI_Get_version(version, subversion)
    if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
    return version[0], subversion[0]
end

------------------------------------------------------------------------------
-- Communicators.
------------------------------------------------------------------------------

_M.comm_world = MPI_COMM_WORLD
local rank = int_1()
function _M.rank(comm)
    local err = C.MPI_Comm_rank(comm, rank)
    if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
    return rank[0]
end

local size = int_1()
function _M.size(comm)
    local err = C.MPI_Comm_size(comm, size)
    if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
    return size[0]
end

------------------------------------------------------------------------------
-- Point-to-point communication.
------------------------------------------------------------------------------

local MPI_DATATYPE_NULL = ffi.cast(MPI_Datatype, C.MPI_DATATYPE_NULL)

function _M.send(buf, count, datatype, dest, tag, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  if dest == nil then dest = C.MPI_PROC_NULL end
  local err = C.MPI_Send(buf, count, datatype, dest, tag, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.recv(buf, count, datatype, source, tag, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  if source == nil then source = C.MPI_PROC_NULL end
  local status = MPI_STATUS_IGNORE
  local err = C.MPI_Recv(buf, count, datatype, source, tag, comm, status)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if dest == nil then dest = C.MPI_PROC_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  if source == nil then source = C.MPI_PROC_NULL end
  local status = MPI_STATUS_IGNORE
  local err = C.MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  if dest == nil then dest = C.MPI_PROC_NULL end
  if source == nil then source = C.MPI_PROC_NULL end
  local status = MPI_STATUS_IGNORE
  local err = C.MPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

------------------------------------------------------------------------------
-- Collective communication.
------------------------------------------------------------------------------

function _M.barrier(comm)
  local err = C.MPI_Barrier(comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.bcast(buf, count, datatype, root, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  local err = C.MPI_Bcast(buf, count, datatype, root, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

_M.in_place = ffi.cast("void *", C.MPI_IN_PLACE)

function _M.gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm)
  if sendtype == nil then sendtype = MPI_DATATYPE_NULL end
  if recvtype == nil then recvtype = MPI_DATATYPE_NULL end
  local err = C.MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.reduce(sendbuf, recvbuf, count, datatype, op, root, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  local err = C.MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.allreduce(sendbuf, recvbuf, count, datatype, op, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  local err = C.MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.scan(sendbuf, recvbuf, count, datatype, op, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  local err = C.MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.exscan(sendbuf, recvbuf, count, datatype, op, comm)
  if datatype == nil then datatype = MPI_DATATYPE_NULL end
  local err = C.MPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

_M.max    = ffi.cast(MPI_Op, C.MPI_MAX)
_M.maxloc = ffi.cast(MPI_Op, C.MPI_MAXLOC)
_M.min    = ffi.cast(MPI_Op, C.MPI_MIN)
_M.minloc = ffi.cast(MPI_Op, C.MPI_MINLOC)
_M.sum    = ffi.cast(MPI_Op, C.MPI_SUM)
_M.prod   = ffi.cast(MPI_Op, C.MPI_PROD)
_M.land   = ffi.cast(MPI_Op, C.MPI_LAND)
_M.band   = ffi.cast(MPI_Op, C.MPI_BAND)
_M.lor    = ffi.cast(MPI_Op, C.MPI_LOR)
_M.bor    = ffi.cast(MPI_Op, C.MPI_BOR)
_M.lxor   = ffi.cast(MPI_Op, C.MPI_LXOR)
_M.bxor   = ffi.cast(MPI_Op, C.MPI_BXOR)

------------------------------------------------------------------------------
-- Datatypes.
------------------------------------------------------------------------------

local function type_free(datatype)
  if finalized() then return end
  local datatype = MPI_Datatype_1(datatype)
  local err = C.MPI_Type_free(datatype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function _M.type_contiguous(count, datatype)
  local newtype = MPI_Datatype_1()
  local err = C.MPI_Type_contiguous(count, datatype, newtype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return ffi.gc(newtype[0], type_free)
end

function _M.type_vector(count, blocklength, stride, datatype)
  local newtype = MPI_Datatype_1()
  local err = C.MPI_Type_vector(count, blocklength, stride, datatype, newtype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return ffi.gc(newtype[0], type_free)
end

function _M.type_create_indexed_block(count, blocklength, displacements, datatype)
  local newtype = MPI_Datatype_1()
  local err = C.MPI_Type_create_indexed_block(count, blocklength, displacements, datatype, newtype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return ffi.gc(newtype[0], type_free)
end

function _M.type_indexed(count, blocklengths, displacements, datatype)
  local newtype = MPI_Datatype_1()
  local err = C.MPI_Type_indexed(count, blocklengths, displacements, datatype, newtype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return ffi.gc(newtype[0], type_free)
end

function _M.type_create_struct(count, blocklengths, displacements, datatypes)
  local newtype = MPI_Datatype_1()
  local err = C.MPI_Type_create_struct(count, blocklengths, displacements, datatypes, newtype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return ffi.gc(newtype[0], type_free)
end

function _M.type_create_resized(datatype, lb, extent)
  local newtype = MPI_Datatype_1()
  local err = C.MPI_Type_create_resized(datatype, lb, extent, newtype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return ffi.gc(newtype[0], type_free)
end

local Datatype = {}

function Datatype.commit(datatype)
  local datatype = MPI_Datatype_1(datatype)
  local err = C.MPI_Type_commit(datatype)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
end

function Datatype.get_extent(datatype)
  local lb, extent = MPI_Aint_1(), MPI_Aint_1()
  local err = C.MPI_Type_get_extent(datatype, lb, extent)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return lb[0], extent[0]
end

--ffi.metatype("struct _MPI_Datatype", {__index = Datatype})

_M.char       = ffi.cast(MPI_Datatype, C.MPI_CHAR)
_M.wchar      = ffi.cast(MPI_Datatype, C.MPI_WCHAR)
_M.schar      = ffi.cast(MPI_Datatype, C.MPI_SIGNED_CHAR)
_M.uchar      = ffi.cast(MPI_Datatype, C.MPI_UNSIGNED_CHAR)
_M.short      = ffi.cast(MPI_Datatype, C.MPI_SHORT)
_M.ushort     = ffi.cast(MPI_Datatype, C.MPI_UNSIGNED_SHORT)
_M.int        = ffi.cast(MPI_Datatype, C.MPI_INT)
_M.uint       = ffi.cast(MPI_Datatype, C.MPI_UNSIGNED)
_M.long       = ffi.cast(MPI_Datatype, C.MPI_LONG)
_M.ulong      = ffi.cast(MPI_Datatype, C.MPI_UNSIGNED_LONG)
_M.llong      = ffi.cast(MPI_Datatype, C.MPI_LONG_LONG)
_M.ullong     = ffi.cast(MPI_Datatype, C.MPI_UNSIGNED_LONG_LONG)
_M.float      = ffi.cast(MPI_Datatype, C.MPI_FLOAT)
_M.double     = ffi.cast(MPI_Datatype, C.MPI_DOUBLE)
_M.short_int  = ffi.cast(MPI_Datatype, C.MPI_SHORT_INT)
_M.int_int    = ffi.cast(MPI_Datatype, C.MPI_2INT)
_M.long_int   = ffi.cast(MPI_Datatype, C.MPI_LONG_INT)
_M.float_int  = ffi.cast(MPI_Datatype, C.MPI_FLOAT_INT)
_M.double_int = ffi.cast(MPI_Datatype, C.MPI_DOUBLE_INT)
_M.byte       = ffi.cast(MPI_Datatype, C.MPI_BYTE)
_M.packed     = ffi.cast(MPI_Datatype, C.MPI_PACKED)

_M.bool       = ffi.cast(MPI_Datatype, C.MPI_C_BOOL)
_M.int8       = ffi.cast(MPI_Datatype, C.MPI_INT8_T)
_M.uint8      = ffi.cast(MPI_Datatype, C.MPI_UINT8_T)
_M.int16      = ffi.cast(MPI_Datatype, C.MPI_INT16_T)
_M.uint16     = ffi.cast(MPI_Datatype, C.MPI_UINT16_T)
_M.int32      = ffi.cast(MPI_Datatype, C.MPI_INT32_T)
_M.uint32     = ffi.cast(MPI_Datatype, C.MPI_UINT32_T)
_M.int64      = ffi.cast(MPI_Datatype, C.MPI_INT64_T)
_M.uint64     = ffi.cast(MPI_Datatype, C.MPI_UINT64_T)
_M.fcomplex   = ffi.cast(MPI_Datatype, C.MPI_C_FLOAT_COMPLEX)
_M.dcomplex   = ffi.cast(MPI_Datatype, C.MPI_C_DOUBLE_COMPLEX)
_M.aint       = ffi.cast(MPI_Datatype, C.MPI_AINT)
_M.offset     = ffi.cast(MPI_Datatype, C.MPI_OFFSET)

------------------------------------------------------------------------------
-- Process topologies.
------------------------------------------------------------------------------

function _M.cart_create(comm, dims, periods, reorder)
  local ndims = #dims
  local dims = int_n(ndims, dims)
  local periods = int_n(ndims, periods)
  local comm_cart = MPI_Comm_1()
  local err = C.MPI_Cart_create(comm, ndims, dims, periods, reorder, comm_cart)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  if comm_cart[0] == MPI_COMM_NULL then return end
  return ffi.gc(comm_cart[0], comm_free)
end

function _M.cart_get(comm)
  local ndims = int_1()
  local err = C.MPI_Cartdim_get(comm, ndims)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  ndims = ndims[0]
  local dims_buf = int_n(ndims)
  local periods_buf = int_n(ndims)
  local coords_buf = int_n(ndims)
  local err = C.MPI_Cart_get(comm, ndims, dims_buf, periods_buf, coords_buf)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  local dims, periods, coords = {}, {}, {}
  for i = 1, ndims do dims[i], periods[i], coords[i] = dims_buf[i-1], periods_buf[i-1] ~= 0, coords_buf[i-1] end
  return dims, periods, coords
end

function _M.cart_coords(comm, rank)
  local ndims = int_1()
  local err = C.MPI_Cartdim_get(comm, ndims)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  ndims = ndims[0]
  local coords_buf = int_n(ndims)
  local err = C.MPI_Cart_coords(comm, rank, ndims, coords_buf)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  local coords = {}
  for i = 1, ndims do coords[i] = coords_buf[i-1] end
  return coords
end

function _M.cart_rank(comm, coords)
  local ndims = int_1()
  local err = C.MPI_Cartdim_get(comm, ndims)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  ndims = ndims[0]
  local coords = int_n(ndims, coords)
  local rank = int_1()
  local err = C.MPI_Cart_rank(comm, coords, rank)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return rank[0]
end

function _M.cart_shift(comm, direction, disp)
  local rank_source, rank_dest = int_1(), int_1()
  local err = C.MPI_Cart_shift(comm, direction, disp, rank_source, rank_dest)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end
  return (rank_source[0] ~= C.MPI_PROC_NULL and rank_source[0] or nil), (rank_dest[0] ~=  C.MPI_PROC_NULL and rank_dest[0] or nil)
end

-- get the allocated local thread id and thread size, used for gpu alloc
function _M.get_local_thread_rank_size(comm)
  local node_name = char_n(C.MPI_MAX_PROCESSOR_NAME)
  local name_len = int_1()
  local err = C.MPI_Get_processor_name(node_name, name_len)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end

  local all_node_name = char_n(_M.size(comm) * C.MPI_MAX_PROCESSOR_NAME)
  err = C.MPI_Allgather(node_name, C.MPI_MAX_PROCESSOR_NAME, _M.char, all_node_name, C.MPI_MAX_PROCESSOR_NAME, _M.char, comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end

  local count = 0
  local node_list = {}

  for i = 1, _M.size(comm) do
    local this_node = ffi.string(all_node_name + (i-1)*C.MPI_MAX_PROCESSOR_NAME, C.MPI_MAX_PROCESSOR_NAME)
    if node_list[this_node] == nil then
      node_list[this_node] = count
      count = count + 1
    end
  end

  local sub_comm = MPI_Comm_1()
  local color = node_list[ffi.string(node_name, C.MPI_MAX_PROCESSOR_NAME)]
  assert(color ~= nil)
  err = C.MPI_Comm_split(comm, color, _M.rank(comm), sub_comm)
  if err ~= C.MPI_SUCCESS then return error(error_string(err)) end

  local sub_rank = _M.rank(sub_comm[0])
  local sub_size = _M.size(sub_comm[0])

  return sub_rank, sub_size
end

return _M
