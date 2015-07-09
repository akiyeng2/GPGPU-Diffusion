#define getIndex(i, j) (i) * width + (j)

__kernel void diffusion(__global float *currentGrid, int width) {
	int tx = get_global_id(0);
	int ty = get_global_id(1);
	
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	int bx = get_group_id(0);
	int by = get_group_id(1);
	
	if(tx > 0 && tx < width - 1 && ty > 0 && ty < width - 1) {
		float left = currentGrid[getIndex(tx - 1, ty)];
		float right = currentGrid[getIndex(tx + 1, ty)];
		float up = currentGrid[getIndex(tx, ty - 1)];
		float down = currentGrid[getIndex(tx, ty + 1)];
		currentGrid[getIndex(tx, ty)] = (left + right + up + down) * 0.25;
	}
}
