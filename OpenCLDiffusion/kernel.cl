//Convert a 2d array index to a 1d array index
int getIdx(int i, int j, int width) {
	return i * width + j;
}

__kernel void diffuse(__global float *grid, __local float *temp, int width, int blockSize) {


	int bx = get_group_id(0);
	int by = get_group_id(1);
	
	int tx = get_local_id(0);
	int ty = get_local_id(1);
   
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	

	temp[getIdx(tx, ty, blockSize)] = grid[getIdx(gx, gy, width)];
	barrier(CLK_LOCAL_MEM_FENCE);

	if(gx > 0 && gx < width - 1 && gy > 0 && gy < width - 1) {
		float left, right, up, down, value;
		

		if(blockSize > 1) {
			if(tx == 0) {
				left = grid[getIdx(gx - 1, gy, width)];
				right = temp[getIdx(tx + 1, ty, blockSize)];
			} else if(tx == blockSize - 1) {
				right = grid[getIdx(gx + 1, gy, width)];
				left = temp[getIdx(tx - 1, ty, blockSize)];
			} else {
				left = temp[getIdx(tx - 1, ty, blockSize)];
				right = temp[getIdx(tx + 1, ty, blockSize)];
			}

			if(ty == 0) {
				up = grid[getIdx(gx, gy - 1, width)];
				down = temp[getIdx(tx, ty + 1, blockSize)];
			} else if (ty == blockSize - 1) {
				down = grid[getIdx(gx, gy + 1, width)];
				up = temp[getIdx(tx, ty - 1, blockSize)];
			} else {
				down = temp[getIdx(tx, ty + 1, blockSize)];
				up = temp[getIdx(tx, ty - 1, blockSize)];
			}
			
		} else {
			left = grid[getIdx(gx - 1, gy, width)];
			right = grid[getIdx(gx + 1, gy, width)];
			up = grid[getIdx(gx, gy - 1, width)];
			down = grid[getIdx(gx, gy + 1, width)];
		}
		value = (left + right + up + down) * 0.25;
		temp[getIdx(tx, ty, blockSize)] = value;
		grid[getIdx(gx, gy, width)] = value;
	}
	
}

