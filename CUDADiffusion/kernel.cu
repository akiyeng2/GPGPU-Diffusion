#define getIdx(i, j) ((i) * width + (j))

__global__ void diffuse(float *grid, int width, int blockSize) {
	 
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
   
	int gx = bx * blockSize + tx;
	int gy = by * blockSize + ty;	

	const int BLOCK_SIZE = blockSize;

	__shared__ float temp[4][4];
	
	temp[tx][ty] = grid[getIdx(gx, gy)];

	__syncthreads();
	
	if(gx > 0 && gx < width - 1 && gy > 0 && gy < width - 1) {
		float left, right, up, down;
		
		if(tx == 0) {
			left = grid[getIdx(gx - 1, gy)];
			right = temp[tx + 1][ty];
		} else if(tx == blockSize - 1) {
			right = grid[getIdx(gx + 1, gy)];
			left = temp[tx - 1][ty];
		} else {
			left = temp[tx - 1][ty];
			right = temp[tx + 1][ty];
		}

		if(ty == 0) {
			up = grid[getIdx(gx, gy - 1)];
			down = temp[tx][ty + 1];
		} else if (ty == blockSize - 1) {
			down = grid[getIdx(gx, gy + 1)];
			up = grid[getIdx(tx, ty - 1)];
		} else {
			down = grid[getIdx(tx, ty + 1)];
			up = grid[getIdx(tx, ty - 1)];
		}
			
		grid[getIdx(gx, gy)] = (left + right + up + down) * 0.25;
	}
	
}

