#define getIndex(i, j) (i) * width + (j)

#define BLOCK_SIZE 4
#define TILE_SIZE 4


__kernel void diffusion(__global float *currentGrid, int width) {
	int tx = get_global_id(0);
	int ty = get_global_id(1);
	
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	int bx = get_group_id(0);
	int by = get_group_id(1);
	
//	int begin = width * BLOCK_SIZE * by;
//	int end = begin + width - 1;
//	int step = BLOCK_SIZE;
//	
//	float average = 100;
//	for(int i = begin; i < end; i += step) {
//		__local float subGrid[BLOCK_SIZE][BLOCK_SIZE];
//		subGrid[ly][lx] = currentGrid[i + width * ty + tx];
//		barrier(CLK_LOCAL_MEM_FENCE);
//		if(tx > 0 && tx < width - 1 && ty > 0 && ty < width - 1) {
//			float left, right, up, down;
//			if(lx == 0) {
//				left = currentGrid[getIndex(tx - 1, ty)];
//			} else if(lx == TILE_SIZE) {
//				right = currentGrid[getIndex(tx - 1, ty)];
//			} else {
//				left = subGrid[lx - 1][ly];
//				right = subGrid[lx + 1][ly];
//			}
//			
//			if(ly == 0) {
//				up = currentGrid[getIndex(tx, ty - 1)];
//			} else if(ly == TILE_SIZE) {
//				down = currentGrid[getIndex(tx, ty - 1)];
//			} else {
//				up = subGrid[lx][ly - 1];
//				down = subGrid[lx + 1][ly + 1];
//			}
//			
//			average = (left + right + up + down) * 0.25;
//			barrier(CLK_LOCAL_MEM_FENCE);
//		}
//		
//	}
//	
//	currentGrid[getIndex(tx, ty)] = average;
	
	
	if(tx > 0 && tx < width - 1 && ty > 0 && ty < width - 1) {
		float left = currentGrid[getIndex(tx - 1, ty)];
		float right = currentGrid[getIndex(tx + 1, ty)];
		float up = currentGrid[getIndex(tx, ty - 1)];
		float down = currentGrid[getIndex(tx, ty + 1)];
		currentGrid[getIndex(tx, ty)] = (left + right + up + down) * 0.25;
	}
}
