#ifdef __OPENCL_C_VERSION__ // make safe to include in resource file to enforce dependency

#ifndef CL_FLT_EPSILON
#define CL_FLT_EPSILON 1.1920928955078125e-7
#endif

#if MAX_PRIM_PER_BATCH == 64u
	#define BIN_TYPE ulong
#elif MAX_PRIM_PER_BATCH == 32u
	#define BIN_TYPE uint
#else
	#error "MAX_PRIM_PER_BATCH != 32u OR 64u"
#endif

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
	char dimx[4][4];
	ulong sel;
	uint fbp, zbp, bw;
	uint fm, zm;
	uchar4 fog; // rgb
	uchar aref, afix;
	uchar ta0, ta1;
	uint tbp[7], tbw[7];
	int minu, maxu, minv, maxv;
	int lod; // lcm == 1
	int mxl;
	float l; // TEX1.L * -0x10000
	float k; // TEX1.K * 0x10000
	uchar4 clut[256]; // TODO: this could be an index to a separate buffer, it may be the same across several gs_params following eachother
} gs_param;

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
	PSM_PSMZ16S,
	PSM_PSMT8,
	PSM_PSMT4,
	PSM_PSMT8H,
	PSM_PSMT4HL,
	PSM_PSMT4HH,
};

enum GS_TFX
{
	TFX_MODULATE	= 0,
	TFX_DECAL		= 1,
	TFX_HIGHLIGHT	= 2,
	TFX_HIGHLIGHT2	= 3,
	TFX_NONE		= 4,
};

enum GS_CLAMP
{
	CLAMP_REPEAT		= 0,
	CLAMP_CLAMP			= 1,
	CLAMP_REGION_CLAMP	= 2,
	CLAMP_REGION_REPEAT	= 3,
};

enum GS_ZTST
{
	ZTST_NEVER		= 0,
	ZTST_ALWAYS		= 1,
	ZTST_GEQUAL		= 2,
	ZTST_GREATER	= 3,
};

enum GS_ATST
{
	ATST_NEVER		= 0,
	ATST_ALWAYS		= 1,
	ATST_LESS		= 2,
	ATST_LEQUAL		= 3,
	ATST_EQUAL		= 4,
	ATST_GEQUAL		= 5,
	ATST_GREATER	= 6,
	ATST_NOTEQUAL	= 7,
};

enum GS_AFAIL
{
	AFAIL_KEEP		= 0,
	AFAIL_FB_ONLY	= 1,
	AFAIL_ZB_ONLY	= 2,
	AFAIL_RGB_ONLY	= 3,
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

__constant uchar blockTable8[4][8] =
{
	{  0,  1,  4,  5, 16, 17, 20, 21},
	{  2,  3,  6,  7, 18, 19, 22, 23},
	{  8,  9, 12, 13, 24, 25, 28, 29},
	{ 10, 11, 14, 15, 26, 27, 30, 31}
};

__constant uchar blockTable4[8][4] =
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

__constant uchar columnTable8[16][16] =
{
	{   0,   4,  16,  20,  32,  36,  48,  52,	// column 0
	    2,   6,  18,  22,  34,  38,  50,  54 },
	{   8,  12,  24,  28,  40,  44,  56,  60,
	   10,  14,  26,  30,  42,  46,  58,  62 },
	{  33,  37,  49,  53,   1,   5,  17,  21,
	   35,  39,  51,  55,   3,   7,  19,  23 },
	{  41,  45,  57,  61,   9,  13,  25,  29,
	   43,  47,  59,  63,  11,  15,  27,  31 },
	{  96, 100, 112, 116,  64,  68,  80,  84, 	// column 1
	   98, 102, 114, 118,  66,  70,  82,  86 },
	{ 104, 108, 120, 124,  72,  76,  88,  92,
	  106, 110, 122, 126,  74,  78,  90,  94 },
	{  65,  69,  81,  85,  97, 101, 113, 117,
	   67,  71,  83,  87,  99, 103, 115, 119 },
	{  73,  77,  89,  93, 105, 109, 121, 125,
	   75,  79,  91,  95, 107, 111, 123, 127 },
	{ 128, 132, 144, 148, 160, 164, 176, 180,	// column 2
	  130, 134, 146, 150, 162, 166, 178, 182 },
	{ 136, 140, 152, 156, 168, 172, 184, 188,
	  138, 142, 154, 158, 170, 174, 186, 190 },
	{ 161, 165, 177, 181, 129, 133, 145, 149,
	  163, 167, 179, 183, 131, 135, 147, 151 },
	{ 169, 173, 185, 189, 137, 141, 153, 157,
	  171, 175, 187, 191, 139, 143, 155, 159 },
	{ 224, 228, 240, 244, 192, 196, 208, 212,	// column 3
	  226, 230, 242, 246, 194, 198, 210, 214 },
	{ 232, 236, 248, 252, 200, 204, 216, 220,
	  234, 238, 250, 254, 202, 206, 218, 222 },
	{ 193, 197, 209, 213, 225, 229, 241, 245,
	  195, 199, 211, 215, 227, 231, 243, 247 },
	{ 201, 205, 217, 221, 233, 237, 249, 253,
	  203, 207, 219, 223, 235, 239, 251, 255 },
};

__constant ushort columnTable4[16][32] =
{
	{   0,   8,  32,  40,  64,  72,  96, 104,	// column 0
	    2,  10,  34,  42,  66,  74,  98, 106,
	    4,  12,  36,  44,  68,  76, 100, 108,
	    6,  14,  38,  46,  70,  78, 102, 110 },
	{  16,  24,  48,  56,  80,  88, 112, 120,
	   18,  26,  50,  58,  82,  90, 114, 122,
	   20,  28,  52,  60,  84,  92, 116, 124,
	   22,  30,  54,  62,  86,  94, 118, 126 },
	{  65,  73,  97, 105,   1,   9,  33,  41,
	   67,  75,  99, 107,   3,  11,  35,  43,
	   69,  77, 101, 109,   5,  13,  37,  45,
	   71,  79, 103, 111,   7,  15,  39,  47 },
	{  81,  89, 113, 121,  17,  25,  49,  57,
	   83,  91, 115, 123,  19,  27,  51,  59,
	   85,  93, 117, 125,  21,  29,  53,  61,
	   87,  95, 119, 127,  23,  31,  55,  63 },
	{ 192, 200, 224, 232, 128, 136, 160, 168,	// column 1
	  194, 202, 226, 234, 130, 138, 162, 170,
	  196, 204, 228, 236, 132, 140, 164, 172,
	  198, 206, 230, 238, 134, 142, 166, 174 },
	{ 208, 216, 240, 248, 144, 152, 176, 184,
	  210, 218, 242, 250, 146, 154, 178, 186,
	  212, 220, 244, 252, 148, 156, 180, 188,
	  214, 222, 246, 254, 150, 158, 182, 190 },
	{ 129, 137, 161, 169, 193, 201, 225, 233,
	  131, 139, 163, 171, 195, 203, 227, 235,
	  133, 141, 165, 173, 197, 205, 229, 237,
	  135, 143, 167, 175, 199, 207, 231, 239 },
	{ 145, 153, 177, 185, 209, 217, 241, 249,
	  147, 155, 179, 187, 211, 219, 243, 251,
	  149, 157, 181, 189, 213, 221, 245, 253,
	  151, 159, 183, 191, 215, 223, 247, 255 },
	{ 256, 264, 288, 296, 320, 328, 352, 360,	// column 2
	  258, 266, 290, 298, 322, 330, 354, 362,
	  260, 268, 292, 300, 324, 332, 356, 364,
	  262, 270, 294, 302, 326, 334, 358, 366 },
	{ 272, 280, 304, 312, 336, 344, 368, 376,
	  274, 282, 306, 314, 338, 346, 370, 378,
	  276, 284, 308, 316, 340, 348, 372, 380,
	  278, 286, 310, 318, 342, 350, 374, 382 },
	{ 321, 329, 353, 361, 257, 265, 289, 297,
	  323, 331, 355, 363, 259, 267, 291, 299,
	  325, 333, 357, 365, 261, 269, 293, 301,
	  327, 335, 359, 367, 263, 271, 295, 303 },
	{ 337, 345, 369, 377, 273, 281, 305, 313,
	  339, 347, 371, 379, 275, 283, 307, 315,
	  341, 349, 373, 381, 277, 285, 309, 317,
	  343, 351, 375, 383, 279, 287, 311, 319 },
	{ 448, 456, 480, 488, 384, 392, 416, 424,	// column 3
	  450, 458, 482, 490, 386, 394, 418, 426,
	  452, 460, 484, 492, 388, 396, 420, 428,
	  454, 462, 486, 494, 390, 398, 422, 430 },
	{ 464, 472, 496, 504, 400, 408, 432, 440,
	  466, 474, 498, 506, 402, 410, 434, 442,
	  468, 476, 500, 508, 404, 412, 436, 444,
	  470, 478, 502, 510, 406, 414, 438, 446 },
	{ 385, 393, 417, 425, 449, 457, 481, 489,
	  387, 395, 419, 427, 451, 459, 483, 491,
	  389, 397, 421, 429, 453, 461, 485, 493,
	  391, 399, 423, 431, 455, 463, 487, 495 },
	{ 401, 409, 433, 441, 465, 473, 497, 505,
	  403, 411, 435, 443, 467, 475, 499, 507,
	  405, 413, 437, 445, 469, 477, 501, 509,
	  407, 415, 439, 447, 471, 479, 503, 511 },
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

uint BlockNumber8(int x, int y, uint bp, uint bw)
{
	return bp + ((y >> 1) & ~0x1f) * (bw >> 1) + ((x >> 2) & ~0x1f) + blockTable8[(y >> 4) & 3][(x >> 4) & 7];
}

uint BlockNumber4(int x, int y, uint bp, uint bw)
{
	return bp + ((y >> 2) & ~0x1f) * (bw >> 1) + ((x >> 2) & ~0x1f) + blockTable4[(y >> 4) & 7][(x >> 5) & 3];
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

uint PixelAddress8(int x, int y, uint bp, uint bw)
{
	return (BlockNumber8(x, y, bp, bw) << 8) + columnTable8[y & 15][x & 15];
}

uint PixelAddress4(int x, int y, uint bp, uint bw)
{
	return (BlockNumber4(x, y, bp, bw) << 9) + columnTable4[y & 15][x & 31];
}

uint PixelAddress(int x, int y, uint bp, uint bw, uint psm)
{
	switch(psm)
	{
	default:
	case PSM_PSMCT32: 
	case PSM_PSMCT24: 
	case PSM_PSMT8H:
	case PSM_PSMT4HL:
	case PSM_PSMT4HH:
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
	case PSM_PSMT8:
		return PixelAddress8(x, y, bp, bw);
	case PSM_PSMT4:
		return PixelAddress4(x, y, bp, bw);
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

uint ReadFrame(__global uchar* vm, uint addr, uint psm)
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

void WriteFrame(__global uchar* vm, uint addr, uint psm, uint value)
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

bool is16bit(int psm)
{
	return psm < 8 && (psm & 3) >= 2;
}

bool is24bit(int psm)
{
	return psm < 8 && (psm & 3) == 1;
}

bool is32bit(int psm)
{
	return psm < 8 && (psm & 3) == 0;
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

		int4 mask = (int4)(v0->p.xy > v1->p.xy, 0, 0);

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

bool ZTest(uint zs, uint zd)
{ 
	if(ZTEST)
	{
		if(is24bit(ZPSM)) zd &= 0x00ffffff;

		switch(ZTST)
		{
		case ZTST_NEVER:
			return false;
		case ZTST_ALWAYS:
			return true;
		case ZTST_GEQUAL:
			return zs >= zd;
		case ZTST_GREATER:
			return zs > zd;
		}
	}

	return true;
}

bool AlphaTest(int alpha, int aref, uint* fm, uint* zm)
{
	switch(AFAIL)
	{
	case AFAIL_KEEP:
		break;
	case AFAIL_FB_ONLY:
		if(!ZWRITE) return true;
		break;
	case AFAIL_ZB_ONLY:
		if(!FWRITE) return true;
		break;
	case AFAIL_RGB_ONLY:
		if(!ZWRITE && is24bit(FPSM)) return true;
		break;
	}

	uint pass;
	
	switch(ATST)
	{
	case ATST_NEVER:
		pass = false;
		break;
	case ATST_ALWAYS:
		return true;
	case ATST_LESS:
		pass = alpha < aref;
		break;
	case ATST_LEQUAL:
		pass = alpha <= aref;
		break;
	case ATST_EQUAL:
		pass = alpha == aref;
		break;
	case ATST_GEQUAL:
		pass = alpha >= aref;
		break;
	case ATST_GREATER:
		pass = alpha > aref;
		break;
	case ATST_NOTEQUAL:
		pass = alpha != aref;
		break;
	}

	switch(AFAIL)
	{
	case AFAIL_KEEP:
		return pass;
	case AFAIL_FB_ONLY:
		*zm |= pass ? 0 : 0xffffffff;
		break;
	case AFAIL_ZB_ONLY:
		*fm |= pass ? 0 : 0xffffffff;
		break;
	case AFAIL_RGB_ONLY:
		*fm |= pass ? 0 : 0xff000000;
		*zm |= pass ? 0 : 0xffffffff;
		break;
	}

	return true;
}

bool DestAlphaTest(uint fd)
{
	if(DATE)
	{
		if(DATM)
		{
			if(is32bit(FPSM)) return (fd & 0x80000000) != 0;
			if(is16bit(FPSM)) return (fd & 0x00008000) != 0;
		}
		else
		{
			if(is32bit(FPSM)) return (fd & 0x80000000) == 0;
			if(is16bit(FPSM)) return (fd & 0x00008000) == 0;
		}
	}

	return true;
}

int Wrap(int a, int b, int c, int mode)
{
	switch(mode)
	{
	case CLAMP_REPEAT:
		return a & b;
	case CLAMP_CLAMP:
		return clamp(a, 0, c);
	case CLAMP_REGION_CLAMP:
		return clamp(a, b, c);
	case CLAMP_REGION_REPEAT:
		return (a & b) | c;
	}
}

int4 AlphaBlend(int4 c, int afix, uint fd)
{
	if(FWRITE && (ABE || AA1))
	{
		int4 cs = c;
		int4 cd;

		if(ABA != ABB && (ABA == 1 || ABB == 1 || ABC == 1) || ABD == 1)
		{
			if(is32bit(FPSM) || is24bit(FPSM))
			{
				cd.x = fd & 0xff;
				cd.y = (fd >> 8) & 0xff;
				cd.z = (fd >> 16) & 0xff;
				cd.w = fd >> 24;
			}
			else if(is16bit(FPSM))
			{
				cd.x = (fd & 0x001f) << 3;
				cd.y = (fd & 0x03e0) >> 2;
				cd.z = (fd & 0x7c00) >> 7;
				cd.w = (fd & 0x8000) >> 8;
			}
		}

		if(ABA != ABB)
		{
			switch(ABA)
			{
			case 0: break; // c.xyz = cs.xyz;
			case 1: c.xyz = cd.xyz; break;
			case 2: c.xyz = 0; break;
			}

			switch(ABB)
			{
			case 0: c.xyz -= cs.xyz; break;
			case 1: c.xyz -= cd.xyz; break;
			case 2: break;
			}

			if(!(is24bit(FPSM) && ABC == 1))
			{
				int a = 0;

				switch(ABC)
				{
				case 0: a = cs.w; break;
				case 1: a = cd.w; break;
				case 2: a = afix; break;
				}

				c.xyz = c.xyz * a >> 7;
			}

			switch(ABD)
			{
			case 0: c.xyz += cs.xyz; break;
			case 1: c.xyz += cd.xyz; break;
			case 2: break;
			}
		}
		else
		{
			switch(ABD)
			{
			case 0: break;
			case 1: c.xyz = cd.xyz; break;
			case 2: c.xyz = 0; break;
			}
		}

		if(PABE)
		{
			c.xyz = select(cs.xyz, c.xyz, (int3)(cs.w << 24));
		}
	}

	return c;
}

uchar4 Expand24To32(uint rgba, uchar ta0)
{
	uchar4 c;

	c.x = rgba & 0xff;
	c.y = (rgba >> 8) & 0xff;
	c.z = (rgba >> 16) & 0xff;
	c.w = !AEM || (rgba & 0xffffff) != 0 ? ta0 : 0;

	return c;
}

uchar4 Expand16To32(ushort rgba, uchar ta0, uchar ta1)
{
	uchar4 c;

	c.x = (rgba & 0x001f) << 3;
	c.y = (rgba & 0x03e0) >> 2;
	c.z = (rgba & 0x7c00) >> 7;
	c.w = !AEM || (rgba & 0x7fff) != 0 ? ((rgba & 0x8000) ? ta1 : ta0) : 0;

	return c;
}

int4 ReadTexel(__global uchar* vm, int x, int y, int level, __global gs_param* pb)
{
	uchar4 c;

	uint addr = PixelAddress(x, y, pb->tbp[level], pb->tbw[level], TPSM);

	__global ushort* vm16 = (__global ushort*)vm;
	__global uint* vm32 = (__global uint*)vm;

	switch(TPSM)
	{
	default:
	case PSM_PSMCT32: 
	case PSM_PSMZ32:
		c = ((__global uchar4*)vm)[addr];
		break;
	case PSM_PSMCT24: 
	case PSM_PSMZ24: 
		c = Expand24To32(vm32[addr], pb->ta0);
		break;
	case PSM_PSMCT16: 
	case PSM_PSMCT16S: 
	case PSM_PSMZ16: 
	case PSM_PSMZ16S: 
		c = Expand16To32(vm16[addr], pb->ta0, pb->ta1);
		break;
	case PSM_PSMT8:
		c = pb->clut[vm[addr]];
		break;
	case PSM_PSMT4:
		c = pb->clut[(vm[addr] >> ((addr & 1) << 2)) & 0x0f];
		break;
	case PSM_PSMT8H:
		c = pb->clut[vm32[addr] >> 24];
		break;
	case PSM_PSMT4HL:
		c = pb->clut[(vm32[addr] >> 24) & 0x0f];
		break;
	case PSM_PSMT4HH:
		c = pb->clut[(vm32[addr] >> 28) & 0x0f];
		break;
	}

	//printf("[%d %d] %05x %d %d %08x | %v4hhd | %08x\n", x, y, pb->tbp[level], pb->tbw[level], TPSM, addr, c, vm32[addr]);

	return convert_int4(c);
}

__kernel void KERNEL_TFX(
	__global gs_env* env,
	__global uchar* vm,
	__global uchar* tex,
	__global uchar* pb_base, 
	uint pb_start,
	uint prim_start, 
	uint prim_count,
	uint batch_count,
	uint bin_count, // == bin_dim.z * bin_dim.w
	uchar4 bin_dim)
{
	// TODO: try it the bin_index = atomic_inc(&env->bin_counter) way

	uint bin_x = (get_global_id(0) >> BIN_SIZE_BITS) - bin_dim.x;
	uint bin_y = (get_global_id(1) >> BIN_SIZE_BITS) - bin_dim.y;
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

	//

	__global gs_param* pb = (__global gs_param*)(pb_base + pb_start);

	uint x = get_global_id(0);
	uint y = get_global_id(1);

	int2 pi = (int2)(x, y);
	float2 pf = convert_float2(pi);

	if(!NOSCISSOR)
	{
		int4 scissor = pb->scissor;

		if(!all((pi >= scissor.xy) & (pi < scissor.zw)))
		{
			return;
		}
	}

	uint faddr = PixelAddress(x, y, pb->fbp, pb->bw, FPSM);
	uint zaddr = PixelAddress(x, y, pb->zbp, pb->bw, ZPSM);

	uint fd, zd;

	if(RFB) 
	{
		fd = ReadFrame(vm, faddr, FPSM);
	}

	if(ZTEST)
	{
		zd = ReadFrame(vm, zaddr, ZPSM);
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

	// early destination alpha test

	if(!DestAlphaTest(fd))
	{
		return;
	}

	//

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

			uint2 zf;
			float4 t;
			int4 c;

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

				zf = convert_uint2(prim->v[0].p.zw * f.z + prim->v[1].p.zw * f.x + prim->v[2].p.zw * f.y);

				t.xyz = prim->v[0].tc.xyz * f.z + prim->v[1].tc.xyz * f.x + prim->v[2].tc.xyz * f.y;

				if(IIP)
				{
					float4 c0 = convert_float4(prim->v[0].c);
					float4 c1 = convert_float4(prim->v[1].c);
					float4 c2 = convert_float4(prim->v[2].c);

					c = convert_int4(c0 * f.z + c1 * f.x + c2 * f.y);
				}
				else
				{
					c = convert_int4(prim->v[2].c);
				}
			}
			else if(PRIM == GS_SPRITE_CLASS)
			{
				int2 tl = convert_int2_rtp(prim->v[0].p.xy);
				int2 br = convert_int2_rtp(prim->v[1].p.xy);

				if(!all((pi >= tl) & (pi < br)))
				{
					continue;
				}

				zf = convert_uint2(prim->v[1].p.zw); // TODO: send as uint
				
				t.xy = prim->v[0].tc.xy + prim->v[1].tc.xy * (pf - prim->v[0].p.xy);
				t.z = prim->v[0].tc.z;

				c = convert_int4(prim->v[1].c);
			}

			// z test

			uint zs = zf.x;

			if(!ZTest(zs, zd))
			{
				continue;
			}

			// sample texture

			int4 ct;

			if(FB && TFX != TFX_NONE)
			{
				// TODO

				if(0)//if(MMIN)
				{
					// TODO
				}
				else
				{
					int2 uv;

					if(!FST)
					{
						uv = convert_int2(t.xy * (1.0f / t.z));

						if(LTF) uv -= 0x0008;
					}
					else
					{
						uv = convert_int2(t.xy);
					}

					int2 uvf = uv & 0x000f;

					int2 uv0 = uv >> 4;
					int2 uv1 = uv0 + 1;

					uv0.x = Wrap(uv0.x, pb->minu, pb->maxu, WMS);
					uv0.y = Wrap(uv0.y, pb->minv, pb->maxv, WMT);
					uv1.x = Wrap(uv1.x, pb->minu, pb->maxu, WMS);
					uv1.y = Wrap(uv1.y, pb->minv, pb->maxv, WMT);
					
					tex = vm; // TODO: use the texture cache

					int4 c00 = ReadTexel(tex, uv0.x, uv0.y, 0, pb);
					int4 c01 = ReadTexel(tex, uv1.x, uv0.y, 0, pb);
					int4 c10 = ReadTexel(tex, uv0.x, uv1.y, 0, pb);
					int4 c11 = ReadTexel(tex, uv1.x, uv1.y, 0, pb);

					if(LTF)
					{
						c00 = ((c01 - c00) * uvf.x >> 4) + c00;
						c10 = ((c11 - c10) * uvf.x >> 4) + c10;
						c00 = ((c10 - c00) * uvf.y >> 4) + c00;
					}

					ct = c00;
				}
			}

			// alpha tfx

			if(FB)
			{
				if(TCC)
				{
					switch(TFX)
					{
					case TFX_MODULATE:
						c.w = clamp(ct.w * c.w >> 7, 0, 0xff);
						break;
					case TFX_DECAL:
						c.w = ct.w;
						break;
					case TFX_HIGHLIGHT:
						c.w = clamp(ct.w + c.w, 0, 0xff);
						break;
					case TFX_HIGHLIGHT2:
						c.w = ct.w;
						break;
					}
				}

				if(AA1)
				{
					if(!ABE || c.w == 0x80)
					{
						// TODO: c.w = coverage; // coverage 0x80 at 100%
					}
				}
			}

			// read mask (read once outside the loop if alpha test does not modify, not sure if it does not get optimized there anyway)

			uint fm = pb->fm;
			uint zm = pb->zm;

			// alpha test

			if(!AlphaTest(c.w, pb->aref, &fm, &zm))
			{
				continue;
			}

			// all tests done, we have a new output

			fragments++;

			// write z

			if(ZWRITE)
			{
				zd = bitselect(zs, zd, zm);
			}

			// rgb tfx

			if(FWRITE)
			{
				switch(TFX)
				{
				case TFX_MODULATE:
					c.xyz = clamp(ct.xyz * c.xyz >> 7, 0, 0xff);
					break;
				case TFX_DECAL:
					c.xyz = ct.xyz;
					break;
				case TFX_HIGHLIGHT:
				case TFX_HIGHLIGHT2:					
					c.xyz = clamp((ct.xyz * c.xyz >> 7) + c.w, 0, 0xff);
					break;
				}
			}

			// fog

			if(FWRITE && FGE)
			{
				int fog = (int)zf.y;

				c.xyz = (c.xyz * fog >> 8) + (convert_int4(pb->fog).xyz * (int3)(0xff - fog) >> 8);				
			}

			// alpha blend

			c = AlphaBlend(c, pb->afix, fd);

			// write frame

			if(FWRITE)
			{
				if(DTHE && is16bit(FPSM))
				{
					// TODO: c += pb->dimx[y & 3]
				}

				c = COLCLAMP ? clamp(c, 0, 0xff) : (c & 0xff);
				
				if(FBA && !is24bit(FPSM))
				{
					c.w |= 0x80;
				}

				uint fs;

				if(is32bit(FPSM))
				{
					fs = (c.w << 24) | (c.z << 16) | (c.y << 8) | c.x;
				}
				else if(is24bit(FPSM))
				{
					fs = (c.z << 16) | (c.y << 8) | c.x;
				}
				else if(is16bit(FPSM))
				{
					fs = ((c.w & 0x80) << 8) | ((c.z & 0xf8) << 7) | ((c.y & 0xf8) << 2) | (c.x >> 3);
				}

				fd = RFB ? bitselect(fs, fd, fm) : fs;

				// dest alpha test for the next loop

				if(!DestAlphaTest(fd))
				{
					prim_index = prim_count; // game over

					break;
				}
			}
		}

		bin += bin_count;
		bin_value = *bin;
	}

	if(fragments > 0)
	{
		// TODO: write color/z to faddr/zaddr (if 16x16 was cached, barrier local mem, swizzle back to its place)

		// TODO if(fm/zm != 0xffffffff) or whatever masks the output completely for the pixel format)

		if(ZWRITE)
		{
			WriteFrame(vm, zaddr, ZPSM, zd);
		}

		if(FWRITE)
		{
			WriteFrame(vm, faddr, FPSM, fd);
			//WriteFrame(vm, faddr, FPSM, 0xff202020 * fragments);
		}
	}
}

#endif

#endif
