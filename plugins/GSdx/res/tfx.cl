#ifdef __OPENCL_C_VERSION__ // make safe to include in resource file to enforce dependency

#ifndef CL_FLT_EPSILON
#define CL_FLT_EPSILON 1.1920928955078125e-7
#endif

enum GS_PRIM_CLASS
{
	GS_POINT_CLASS,
	GS_LINE_CLASS,
	GS_TRIANGLE_CLASS,
	GS_SPRITE_CLASS
};

enum GS_PSM_TARGET
{
	PSM_PSMCT32,
	PSM_PSMCT24,
	PSM_PSMCT16,
	PSM_PSMCT16S,
	PSM_PSMZ32,
	PSM_PSMZ24,
	PSM_PSMZ16,
	PSM_PSMZ16S
};

__constant uchar blockTable32[4][8] =
{
	{  0,  1,  4,  5, 16, 17, 20, 21},
	{  2,  3,  6,  7, 18, 19, 22, 23},
	{  8,  9, 12, 13, 24, 25, 28, 29},
	{ 10, 11, 14, 15, 26, 27, 30, 31}
};

__constant uchar blockTable32Z[4][8] =
{
	{ 24, 25, 28, 29,  8,  9, 12, 13},
	{ 26, 27, 30, 31, 10, 11, 14, 15},
	{ 16, 17, 20, 21,  0,  1,  4,  5},
	{ 18, 19, 22, 23,  2,  3,  6,  7}
};

__constant uchar blockTable16[8][4] =
{
	{  0,  2,  8, 10 },
	{  1,  3,  9, 11 },
	{  4,  6, 12, 14 },
	{  5,  7, 13, 15 },
	{ 16, 18, 24, 26 },
	{ 17, 19, 25, 27 },
	{ 20, 22, 28, 30 },
	{ 21, 23, 29, 31 }
};

__constant uchar blockTable16S[8][4] =
{
	{  0,  2, 16, 18 },
	{  1,  3, 17, 19 },
	{  8, 10, 24, 26 },
	{  9, 11, 25, 27 },
	{  4,  6, 20, 22 },
	{  5,  7, 21, 23 },
	{ 12, 14, 28, 30 },
	{ 13, 15, 29, 31 }
};

__constant uchar blockTable16Z[8][4] =
{
	{ 24, 26, 16, 18 },
	{ 25, 27, 17, 19 },
	{ 28, 30, 20, 22 },
	{ 29, 31, 21, 23 },
	{  8, 10,  0,  2 },
	{  9, 11,  1,  3 },
	{ 12, 14,  4,  6 },
	{ 13, 15,  5,  7 }
};

__constant uchar blockTable16SZ[8][4] =
{
	{ 24, 26,  8, 10 },
	{ 25, 27,  9, 11 },
	{ 16, 18,  0,  2 },
	{ 17, 19,  1,  3 },
	{ 28, 30, 12, 14 },
	{ 29, 31, 13, 15 },
	{ 20, 22,  4,  6 },
	{ 21, 23,  5,  7 }
};

__constant uchar columnTable32[8][8] =
{
	{  0,  1,  4,  5,  8,  9, 12, 13 },
	{  2,  3,  6,  7, 10, 11, 14, 15 },
	{ 16, 17, 20, 21, 24, 25, 28, 29 },
	{ 18, 19, 22, 23, 26, 27, 30, 31 },
	{ 32, 33, 36, 37, 40, 41, 44, 45 },
	{ 34, 35, 38, 39, 42, 43, 46, 47 },
	{ 48, 49, 52, 53, 56, 57, 60, 61 },
	{ 50, 51, 54, 55, 58, 59, 62, 63 },
};

__constant uchar columnTable16[8][16] =
{
	{   0,   2,   8,  10,  16,  18,  24,  26,
	    1,   3,   9,  11,  17,  19,  25,  27 },
	{   4,   6,  12,  14,  20,  22,  28,  30,
	    5,   7,  13,  15,  21,  23,  29,  31 },
	{  32,  34,  40,  42,  48,  50,  56,  58,
	   33,  35,  41,  43,  49,  51,  57,  59 },
	{  36,  38,  44,  46,  52,  54,  60,  62,
	   37,  39,  45,  47,  53,  55,  61,  63 },
	{  64,  66,  72,  74,  80,  82,  88,  90,
	   65,  67,  73,  75,  81,  83,  89,  91 },
	{  68,  70,  76,  78,  84,  86,  92,  94,
	   69,  71,  77,  79,  85,  87,  93,  95 },
	{  96,  98, 104, 106, 112, 114, 120, 122,
	   97,  99, 105, 107, 113, 115, 121, 123 },
	{ 100, 102, 108, 110, 116, 118, 124, 126,
	  101, 103, 109, 111, 117, 119, 125, 127 },
};

uint BlockNumber32(int x, int y, uint bp, uint bw)
{
	return bp + (y & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable32[(y >> 3) & 3][(x >> 3) & 7];
}

uint BlockNumber16(int x, int y, uint bp, uint bw)
{
	return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16[(y >> 3) & 7][(x >> 4) & 3];
}

uint BlockNumber16S(int x, int y, uint bp, uint bw)
{
	return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16S[(y >> 3) & 7][(x >> 4) & 3];
}

uint BlockNumber32Z(int x, int y, uint bp, uint bw)
{
	return bp + (y & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable32Z[(y >> 3) & 3][(x >> 3) & 7];
}

uint BlockNumber16Z(int x, int y, uint bp, uint bw)
{
	return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16Z[(y >> 3) & 7][(x >> 4) & 3];
}

uint BlockNumber16SZ(int x, int y, uint bp, uint bw)
{
	return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16SZ[(y >> 3) & 7][(x >> 4) & 3];
}

uint PixelAddress32(int x, int y, uint bp, uint bw)
{
	return (BlockNumber32(x, y, bp, bw) << 6) + columnTable32[y & 7][x & 7];
}

uint PixelAddress16(int x, int y, uint bp, uint bw)
{
	return (BlockNumber16(x, y, bp, bw) << 7) + columnTable16[y & 7][x & 15];
}

uint PixelAddress16S(int x, int y, uint bp, uint bw)
{
	return (BlockNumber16S(x, y, bp, bw) << 7) + columnTable16[y & 7][x & 15];
}

uint PixelAddress32Z(int x, int y, uint bp, uint bw)
{
	return (BlockNumber32Z(x, y, bp, bw) << 6) + columnTable32[y & 7][x & 7];
}

uint PixelAddress16Z(int x, int y, uint bp, uint bw)
{
	return (BlockNumber16Z(x, y, bp, bw) << 7) + columnTable16[y & 7][x & 15];
}

uint PixelAddress16SZ(int x, int y, uint bp, uint bw)
{
	return (BlockNumber16SZ(x, y, bp, bw) << 7) + columnTable16[y & 7][x & 15];
}

uint PixelAddress(int x, int y, uint bp, uint bw, uint psm)
{
	switch(psm)
	{
	default:
	case PSM_PSMCT32: 
	case PSM_PSMCT24: 
		return PixelAddress32(x, y, bp, bw);
	case PSM_PSMCT16: 
		return PixelAddress16(x, y, bp, bw);
	case PSM_PSMCT16S: 
		return PixelAddress16S(x, y, bp, bw);
	case PSM_PSMZ32: 
	case PSM_PSMZ24: 
		return PixelAddress32Z(x, y, bp, bw);
	case PSM_PSMZ16: 
		return PixelAddress16Z(x, y, bp, bw);
	case PSM_PSMZ16S: 
		return PixelAddress16SZ(x, y, bp, bw);
	}
}

uint TileBlockNumber(int x, int y, uint bp, uint bw, uint psm)
{
	// TODO: replace blockTable with a subset tileTable

	switch(psm)
	{
	default:
	case PSM_PSMCT32: 
	case PSM_PSMCT24: 
		return bp + (y & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable32[(y >> 3) & 2][(x >> 3) & 6];
	case PSM_PSMCT16: 
		return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16[(y >> 3) & 2][(x >> 4) & 3];
	case PSM_PSMCT16S: 
		return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16S[(y >> 3) & 2][(x >> 4) & 3];
	case PSM_PSMZ32: 
	case PSM_PSMZ24: 
		return bp + (y & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable32Z[(y >> 3) & 2][(x >> 3) & 6];
	case PSM_PSMZ16: 
		return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16Z[(y >> 3) & 2][(x >> 4) & 3];
	case PSM_PSMZ16S: 
		return bp + ((y >> 1) & ~0x1f) * bw + ((x >> 1) & ~0x1f) + blockTable16SZ[(y >> 3) & 2][(x >> 4) & 3];
	}
}

uint TilePixelAddress(int x, int y, uint ba, uint psm)
{
	switch(psm)
	{
	default:
	case PSM_PSMCT32: 
	case PSM_PSMCT24: 
	case PSM_PSMZ32: 
	case PSM_PSMZ24: 
		return ((ba + ((y >> 2) & 2) + ((x >> 3) & 1)) << 6) + columnTable32[y & 7][x & 7];
	case PSM_PSMCT16: 
	case PSM_PSMCT16S: 
	case PSM_PSMZ16: 
	case PSM_PSMZ16S: 
		return ((ba + ((y >> 3) & 1)) << 7) + columnTable16[y & 7][x & 15];
	}
}

uint ReadPixel(__global uchar* vm, uint addr, uint psm)
{
	switch(psm)
	{
	default:
	case PSM_PSMCT32: 
	case PSM_PSMCT24: 
	case PSM_PSMZ32: 
	case PSM_PSMZ24: 
		return ((__global uint*)vm)[addr];
	case PSM_PSMCT16: 
	case PSM_PSMCT16S: 
	case PSM_PSMZ16: 
	case PSM_PSMZ16S: 
		return ((__global ushort*)vm)[addr];
	}
}

void WritePixel(__global uchar* vm, uint addr, uint psm, uint value)
{
	switch(psm)
	{
	default:
	case PSM_PSMCT32: 
	case PSM_PSMZ32:
	case PSM_PSMCT24: 
	case PSM_PSMZ24: 
		((__global uint*)vm)[addr] = value; 
		break;
	case PSM_PSMCT16: 
	case PSM_PSMCT16S: 
	case PSM_PSMZ16: 
	case PSM_PSMZ16S: 
		((__global ushort*)vm)[addr] = (ushort)value;
		break;
	}
}

#ifdef PRIM

int GetVertexPerPrim(int prim_class)
{
	switch(prim_class)
	{
	default:
	case GS_POINT_CLASS: return 1;
	case GS_LINE_CLASS: return 2;
	case GS_TRIANGLE_CLASS: return 3;
	case GS_SPRITE_CLASS: return 2;
	}
}

#define VERTEX_PER_PRIM GetVertexPerPrim(PRIM)

#endif

#if MAX_PRIM_PER_BATCH == 64u
	#define BIN_TYPE ulong
#elif MAX_PRIM_PER_BATCH == 32u
	#define BIN_TYPE uint
#else
	#error "MAX_PRIM_PER_BATCH != 32u OR 64u"
#endif

/*
typedef struct
{
	float2 st;
	uchar4 rgba;
	float q;
	ushort2 xy;
	uint z;
	ushort2 uv;
	uint fog;
} gs_vertex;
*/
typedef struct
{
	union {float4 p; struct {float x, y, z, f;};};
	union {float4 tc; struct {float s, t, q; uchar4 c;};};
} gs_vertex;

typedef struct
{
	gs_vertex v[4];
} gs_prim;

typedef struct
{
	float4 dx, dy;
	float4 zero;
	float4 reject_corner;
} gs_barycentric;

typedef struct
{
	uint batch_counter;
	uint _pad[7];
	struct {uint first, last;} bounds[MAX_BIN_PER_BATCH];
	BIN_TYPE bin[MAX_BIN_COUNT];
	uchar4 bbox[MAX_PRIM_COUNT];
	gs_prim prim[MAX_PRIM_COUNT];
	gs_barycentric barycentric[MAX_PRIM_COUNT];
} gs_env;

typedef struct
{
	int4 scissor;
	int4 bbox;
	int4 rect;
	int4 dimx[4];
	ulong sel;
	uint fbp, zbp, bw;
	uint fm, zm;
	int aref, afix;
	uint fog; // rgb
	ushort minu, maxu;
	ushort minv, maxv;
	int lod; // lcm == 1
	int mxl;
	float l; // TEX1.L * -0x10000
	float k; // TEX1.K * 0x10000
	uchar4 clut[256]; // TODO: this could be an index to a separate buffer, it may be the same across several gs_params following eachother
} gs_param;

#ifdef KERNEL_PRIM

__kernel void KERNEL_PRIM(
	__global gs_env* env,
	__global uchar* vb_base, 
	__global uchar* ib_base, 
	uint vb_start,
	uint ib_start)
{
	size_t prim_index = get_global_id(0);

	__global gs_vertex* vb = (__global gs_vertex*)(vb_base + vb_start);
	__global uint* ib = (__global uint*)(ib_base + ib_start);
	__global gs_prim* prim = &env->prim[prim_index];
	
	ib += prim_index * VERTEX_PER_PRIM;

	int2 pmin, pmax;

	if(PRIM == GS_POINT_CLASS)
	{
		pmin = pmax = convert_int2_rte(vb[ib[0]].p.xy);
	}
	else if(PRIM == GS_LINE_CLASS)
	{
		int2 p0 = convert_int2_rte(vb[ib[0]].p.xy);
		int2 p1 = convert_int2_rte(vb[ib[1]].p.xy);

		pmin = min(p0, p1);
		pmax = max(p0, p1);
	}
	else if(PRIM == GS_TRIANGLE_CLASS)
	{
		__global gs_vertex* v0 = &vb[ib[0]];
		__global gs_vertex* v1 = &vb[ib[1]];
		__global gs_vertex* v2 = &vb[ib[2]];

		int2 p0 = convert_int2_rtp(v0->p.xy);
		int2 p1 = convert_int2_rtp(v1->p.xy);
		int2 p2 = convert_int2_rtp(v2->p.xy);

		pmin = min(min(p0, p1), p2);
		pmax = max(max(p0, p1), p2);

		prim->v[0].p = v0->p;
		prim->v[0].tc = v0->tc;
		prim->v[1].p = v1->p;
		prim->v[1].tc = v1->tc;
		prim->v[2].p = v2->p;
		prim->v[2].tc = v2->tc;

		float4 dp0 = v1->p - v0->p;
		float4 dp1 = v0->p - v2->p;
		float4 dp2 = v2->p - v1->p;

		float cp = dp0.x * dp1.y - dp0.y * dp1.x;

		if(cp != 0.0f)
		{
			float cp_rcp = 1.0f / cp;// native_recip(cp);

			float2 u = dp0.xy * cp_rcp;
			float2 v = -dp1.xy * cp_rcp;

			// v0 has the (0, 0, 1) barycentric coord, v1: (0, 1, 0), v2: (1, 0, 0)

			gs_barycentric b;

			b.dx = (float4)(-v.y, u.y, v.y - u.y, v0->p.x);
			b.dy = (float4)(v.x, -u.x, u.x - v.x, v0->p.y);

			dp0.xy = dp0.xy * sign(cp);
			dp1.xy = dp1.xy * sign(cp);
			dp2.xy = dp2.xy * sign(cp);

			b.zero.x = (dp1.y < 0 || dp1.y == 0 && dp1.x > 0) ? CL_FLT_EPSILON : 0;
			b.zero.y = (dp0.y < 0 || dp0.y == 0 && dp0.x > 0) ? CL_FLT_EPSILON : 0;
			b.zero.z = (dp2.y < 0 || dp2.y == 0 && dp2.x > 0) ? CL_FLT_EPSILON : 0;

			// any barycentric(reject_corner) < 0, tile outside the triangle

			b.reject_corner.x = 0.0f + max(max(max(0.0f, b.dx.x), b.dy.x), b.dx.x + b.dy.x) * BIN_SIZE;
			b.reject_corner.y = 0.0f + max(max(max(0.0f, b.dx.y), b.dy.y), b.dx.y + b.dy.y) * BIN_SIZE;
			b.reject_corner.z = 1.0f + max(max(max(0.0f, b.dx.z), b.dy.z), b.dx.z + b.dy.z) * BIN_SIZE;

			// TODO: accept_corner, at min value, all barycentric(accept_corner) >= 0, tile fully inside, no per pixel hittest needed

			env->barycentric[prim_index] = b;
		}
		else
		{
			// TODO: set b.zero to something that always fails the tests
		}
	}
	else if(PRIM == GS_SPRITE_CLASS)
	{
		__global gs_vertex* v0 = &vb[ib[0]];
		__global gs_vertex* v1 = &vb[ib[1]];

		int2 p0 = convert_int2_rtp(v0->p.xy);
		int2 p1 = convert_int2_rtp(v1->p.xy);

		pmin = min(p0, p1);
		pmax = max(p0, p1);

		int4 mask = (int4)(v0->p.xy < v1->p.xy, 0, 0);

		prim->v[0].p = select(v0->p, v1->p, mask); // pmin
		prim->v[0].tc = select(v0->tc, v1->tc, mask);
		prim->v[1].p = select(v1->p, v0->p, mask); // pmax
		prim->v[1].tc = select(v1->tc, v0->tc, mask);
		prim->v[1].tc.xy = (prim->v[1].tc.xy - prim->v[0].tc.xy) / (prim->v[1].p.xy - prim->v[0].p.xy);
	}

	int4 pminmax = (int4)(pmin, pmax);

	env->bbox[prim_index] = convert_uchar4_sat(pminmax >> BIN_SIZE_BITS);
}

#endif

#ifdef KERNEL_TILE

int tile_in_triangle(float2 p, gs_barycentric b)
{
	float3 f = b.dx.xyz * (p.x - b.dx.w) + b.dy.xyz * (p.y - b.dy.w) + b.reject_corner.xyz;

	f = select(f, (float3)(0.0f), fabs(f) < (float3)(CL_FLT_EPSILON * 10));

	return all(f >= b.zero.xyz);
}

#if CLEAR == 1

__kernel void KERNEL_TILE(__global gs_env* env)
{
	env->batch_counter = 0;
	env->bounds[get_global_id(0)].first = -1;
	env->bounds[get_global_id(0)].last = 0;
}

#elif MODE < 3

#if MAX_PRIM_PER_BATCH != 32
	#error "MAX_PRIM_PER_BATCH != 32"
#endif

#define MAX_PRIM_PER_GROUP (32u >> MODE)

__kernel void KERNEL_TILE(
	__global gs_env* env,
	uint prim_count,
	uint bin_count, // == bin_dim.z * bin_dim.w
	uchar4 bin_dim)
{
	uint batch_index = get_group_id(2) >> MODE;
	uint prim_start = get_group_id(2) << (5 - MODE);
	uint group_prim_index = get_local_id(2);
	uint bin_index = get_local_id(1) * get_local_size(0) + get_local_id(0);

	__global BIN_TYPE* bin = &env->bin[batch_index * bin_count];
	__global uchar4* bbox = &env->bbox[prim_start];
	__global gs_barycentric* barycentric = &env->barycentric[prim_start];

	__local uchar4 bbox_cache[MAX_PRIM_PER_GROUP];
	__local gs_barycentric barycentric_cache[MAX_PRIM_PER_GROUP];
	__local uint visible[8 << MODE];

	if(get_local_id(2) == 0)
	{
		visible[bin_index] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	uint group_prim_count = min(prim_count - prim_start, MAX_PRIM_PER_GROUP);

	event_t e = async_work_group_copy(bbox_cache, bbox, group_prim_count, 0);

	wait_group_events(1, &e);

	if(PRIM == GS_TRIANGLE_CLASS)
	{
		e = async_work_group_copy((__local float4*)barycentric_cache, (__global float4*)barycentric, group_prim_count * (sizeof(gs_barycentric) / sizeof(float4)), 0);
		
		wait_group_events(1, &e);
	}

	if(group_prim_index < group_prim_count)
	{
		int x = bin_dim.x + get_local_id(0);
		int y = bin_dim.y + get_local_id(1);

		uchar4 r = bbox_cache[group_prim_index];

		uint test = (r.x <= x + 1) & (r.z >= x) & (r.y <= y + 1) & (r.w >= y);

		if(PRIM == GS_TRIANGLE_CLASS && test != 0)
		{
			test &= tile_in_triangle(convert_float2((int2)(x, y) << BIN_SIZE_BITS), barycentric_cache[group_prim_index]);
		}

		atomic_or(&visible[bin_index], test << ((MAX_PRIM_PER_GROUP - 1) - get_local_id(2)));
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(get_local_id(2) == 0)
	{
		#if MODE == 0
		((__global uint*)&bin[bin_index])[0] = visible[bin_index];
		#elif MODE == 1
		((__global ushort*)&bin[bin_index])[1 - (get_group_id(2) & 1)] = visible[bin_index];
		#elif MODE == 2
		((__global uchar*)&bin[bin_index])[3 - (get_group_id(2) & 3)] = visible[bin_index];
		#endif

		if(visible[bin_index] != 0)
		{
			atomic_min(&env->bounds[bin_index].first, batch_index);
			atomic_max(&env->bounds[bin_index].last, batch_index);
		}
	}
}

#elif MODE == 3

__kernel void KERNEL_TILE(
	__global gs_env* env,
	uint prim_count,
	uint batch_count,
	uint bin_count, // == bin_dim.z * bin_dim.w
	uchar4 bin_dim)
{
	__local uchar4 bbox_cache[MAX_PRIM_PER_BATCH];
	__local gs_barycentric barycentric_cache[MAX_PRIM_PER_BATCH];
	__local uint batch_index;

	size_t local_id = get_local_id(0);
	size_t local_size = get_local_size(0);

	while(1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		if(local_id == 0)
		{
			batch_index = atomic_inc(&env->batch_counter);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(batch_index >= batch_count) 
		{
			break;
		}

		uint batch_prim_count = min(prim_count - (batch_index << MAX_PRIM_PER_BATCH_BITS), MAX_PRIM_PER_BATCH);
		
		__global BIN_TYPE* bin = &env->bin[batch_index * bin_count];
		__global uchar4* bbox = &env->bbox[batch_index << MAX_PRIM_PER_BATCH_BITS];
		__global gs_barycentric* barycentric = &env->barycentric[batch_index << MAX_PRIM_PER_BATCH_BITS];

		event_t e = async_work_group_copy(bbox_cache, bbox, batch_prim_count, 0);

		wait_group_events(1, &e);

		if(PRIM == GS_TRIANGLE_CLASS)
		{
			e = async_work_group_copy((__local float4*)barycentric_cache, (__global float4*)barycentric, batch_prim_count * (sizeof(gs_barycentric) / sizeof(float4)), 0);
		
			wait_group_events(1, &e);
		}

		for(uint bin_index = local_id; bin_index < bin_count; bin_index += local_size)
		{
			int y = bin_index / bin_dim.z;
			int x = bin_index - y * bin_dim.z;

			x += bin_dim.x;
			y += bin_dim.y;

			BIN_TYPE visible = 0;

			for(uint i = 0; i < batch_prim_count; i++)
			{
				uchar4 r = bbox_cache[i];

				BIN_TYPE test = (r.x <= x + 1) & (r.z >= x) & (r.y <= y + 1) & (r.w >= y);

				if(PRIM == GS_TRIANGLE_CLASS && test != 0)
				{
					test &= tile_in_triangle(convert_float2((int2)(x, y) << BIN_SIZE_BITS), barycentric_cache[i]);
				}

				visible |= test << ((MAX_PRIM_PER_BATCH - 1) - i);
			}

			bin[bin_index] = visible;

			if(visible != 0)
			{
				atomic_min(&env->bounds[bin_index].first, batch_index);
				atomic_max(&env->bounds[bin_index].last, batch_index);
			}
		}
	}
}

#endif

#endif

#ifdef KERNEL_TFX

__kernel 
	//__attribute__((reqd_work_group_size(16, 16, 1))) 
	void KERNEL_TFX(
	__global gs_env* env,
	__global uchar* vm,
	__global uchar* pb_base, 
	uint pb_start,
	uint prim_start, 
	uint prim_count,
	uint batch_count,
	uint bin_count, // == bin_dim.z * bin_dim.w
	uchar4 bin_dim)
{
	// TODO: try it the bin_index = atomic_inc(&env->bin_counter) way

	uint bin_x = (get_global_offset(0) >> BIN_SIZE_BITS) + get_group_id(0) - bin_dim.x;
	uint bin_y = (get_global_offset(1) >> BIN_SIZE_BITS) + get_group_id(1) - bin_dim.y;
	uint bin_index = bin_y * bin_dim.z + bin_x;

	uint batch_first = env->bounds[bin_index].first;
	uint batch_last = env->bounds[bin_index].last;
	uint batch_start = prim_start >> MAX_PRIM_PER_BATCH_BITS;

	if(batch_last < batch_first)
	{
		return;
	}

	uint skip;
	
	if(batch_start < batch_first)
	{
		uint n = (batch_first - batch_start) * MAX_PRIM_PER_BATCH - (prim_start & (MAX_PRIM_PER_BATCH - 1));

		if(n > prim_count) 
		{
			return;
		}

		skip = 0;
		prim_count -= n;
		batch_start = batch_first;
	}
	else
	{
		skip = prim_start & (MAX_PRIM_PER_BATCH - 1);
		prim_count += skip;
	}

	if(batch_start > batch_last) 
	{
		return;
	}
	
	prim_count = min(prim_count, (batch_last - batch_start + 1) << MAX_PRIM_PER_BATCH_BITS);


	__global gs_param* pb = (__global gs_param*)(pb_base + pb_start);

	uint x = get_global_id(0);
	uint y = get_global_id(1);

	int2 pi = (int2)(x, y);
	float2 pf = convert_float2(pi);

	int4 scissor = pb->scissor;

	if(!all((pi >= scissor.xy) & (pi < scissor.zw)))
	{
		return;
	}

	uint faddr = PixelAddress(x, y, pb->fbp, pb->bw, FPSM);
	uint zaddr = PixelAddress(x, y, pb->zbp, pb->bw, ZPSM);

	uint fd, zd;

	if(ZTEST)
	{
		zd = ReadPixel(vm, zaddr, ZPSM);
	}

	if(RFB) 
	{
		fd = ReadPixel(vm, faddr, FPSM);
	}

/*
	// TODO: lookup top left address of this tile + local offset
	//
	// 32bpp: 8x8 block size, 4 blocks, 1024 bytes
	// 0 1
	// 2 3
	// 16bpp: 16x8 block size, 2 blocks, 512 bytes
	// 0
	// 1
	// linear access in memory, this layout is the same for all formats

	__local uint fbn, zbn;
	__local uchar fb[1024], zb[1024];

	if(get_local_id(0) == 0 && get_local_id(1) == 0)
	{
		fbn = TileBlockNumber(x, y, pb->fbp, pb->bw, FPSM);
		zbn = TileBlockNumber(x, y, pb->fbp, pb->bw, FPSM);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	uint faddr = TilePixelAddress(x, y, fbn, FPSM);
	uint zaddr = TilePixelAddress(x, y, zbn, ZPSM);

	if(RFB)
	{
		event_t e = async_work_group_copy((__local uint4*)fb, (__global uint4*)&vm[fbn << 8], 1024 / sizeof(uint4), 0);
		
		wait_group_events(1, &e);
	}

	if(ZTEST)
	{
		event_t e = async_work_group_copy((__local uint4*)zb, (__global uint4*)&vm[zbn << 8], 1024 / sizeof(uint4), 0);
		
		wait_group_events(1, &e);
	}

	// not sure if faster
*/

	// TODO: early destination alpha test

	uint fragments = 0;

	//__local gs_prim p;

	__global BIN_TYPE* bin = &env->bin[bin_index + batch_start * bin_count]; // TODO: not needed for "one tile case"
	__global gs_prim* prim_base = &env->prim[batch_start << MAX_PRIM_PER_BATCH_BITS];
	__global gs_barycentric* barycentric = &env->barycentric[batch_start << MAX_PRIM_PER_BATCH_BITS];

	BIN_TYPE bin_value = *bin & ((BIN_TYPE)-1 >> skip);

	__local gs_prim prim_cache;

	for(uint prim_index = 0; prim_index < prim_count; prim_index += MAX_PRIM_PER_BATCH)
	{
		while(bin_value != 0)
		{
			uint i = clz(bin_value);

			if(prim_index + i >= prim_count)
			{
				break;
			}

			__global gs_prim* prim = &prim_base[prim_index + i];
			
			bin_value ^= (BIN_TYPE)1 << ((MAX_PRIM_PER_BATCH - 1) - i); // bin_value &= (ulong)-1 >> (i + 1);

			float4 p, t, c;

			 // TODO: do not hittest if we know the tile is fully inside the prim

			if(PRIM == GS_POINT_CLASS)
			{
				// TODO: distance.x < 0.5f || distance.y < 0.5f

				continue;
			}
			else if(PRIM == GS_LINE_CLASS)
			{
				// TODO: find point on line prependicular to (x,y), distance.x < 0.5f || distance.y < 0.5f

				continue;
			}
			else if(PRIM == GS_TRIANGLE_CLASS)
			{
				__global gs_barycentric* b = &barycentric[prim_index + i];

				float3 f = b->dx.xyz * (pf.x - b->dx.w) + b->dy.xyz * (pf.y - b->dy.w) + (float3)(0, 0, 1);

				f = select(f, (float3)(0.0f), fabs(f) < (float3)(CL_FLT_EPSILON * 10));

				if(!all(f >= b->zero.xyz))
				{
					continue;
				}

				float4 c0 = convert_float4(prim->v[0].c);
				float4 c1 = convert_float4(prim->v[1].c);
				float4 c2 = convert_float4(prim->v[2].c);

				p.zw = prim->v[0].p.zw * f.z + prim->v[1].p.zw * f.x + prim->v[2].p.zw * f.y;
				t.xyz = prim->v[0].tc.xyz * f.z + prim->v[1].tc.xyz * f.x + prim->v[2].tc.xyz * f.y;
				c = IIP ? c0 * f.z + c1 * f.x + c2 * f.y : c2;
			}
			else if(PRIM == GS_SPRITE_CLASS)
			{
				int2 tl = convert_int2_rtp(prim->v[0].p.xy);
				int2 br = convert_int2_rtp(prim->v[1].p.xy);

				if(!all((pi >= tl) & (pi < br)))
				{
					continue;
				}
				
				p.zw = prim->v[1].p.zw;
				t.xy = prim->v[0].tc.xy + prim->v[1].tc.xy * (pf - prim->v[0].p.xy);
				t.z = prim->v[0].tc.z;
				c = convert_float4(prim->v[1].c);
			}

			// TODO: tfx(x, y, p, t, c, pb, [in/out] fd, [in/out] zd);

			fd = as_uint(convert_uchar4_sat(c));

			fragments++;
		}

		bin += bin_count;
		bin_value = *bin;
	}

	if(fragments > 0)
	{
		// TODO: write color/z to faddr/zaddr (if 16x16 was cached, barrier local mem, swizzle back to its place)

		if(ZWRITE)
		{
			WritePixel(vm, zaddr, ZPSM, zd);
		}

		if(FWRITE)
		{
			WritePixel(vm, faddr, FPSM, fd);
			//WritePixel(vm, faddr, FPSM, 0xff202020 * fragments);
		}
	}
}

#endif

#endif
