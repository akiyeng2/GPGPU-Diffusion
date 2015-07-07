#define getIndex(i, j) (i) * width + (j)

__kernel void diffusion(__global float *lastGrid, __global float *currentGrid, int width) {
	int tx = get_global_id(0);
	int ty = get_global_id(1);
	float value = 100;
	if(tx > 0 && tx < width - 1 && ty > 0 && ty < width - 1) {
		float left = lastGrid[getIndex(tx - 1, ty)];
		float right = lastGrid[getIndex(tx + 1, ty)];
		float up = lastGrid[getIndex(tx, ty - 1)];
		float down = lastGrid[getIndex(tx, ty + 1)];
		currentGrid[getIndex(tx, ty)] = (left + right + up + down) * 0.25;
	} else {
		currentGrid[getIndex(tx, ty)] = lastGrid[getIndex(tx, ty)];
	}
//	currentGrid[getIndex(tx, ty)] = getIndex(tx, ty);



	
	
}