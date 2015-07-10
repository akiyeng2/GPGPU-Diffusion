#define getIdx(i, j) ((i) * width + (j))

__global__ void diffuse(float *grid, int width, int blockSize) {
 
     int tx = blockIdx.x * blockSize + threadIdx.x;
     int ty = blockIdx.y * blockSize + threadIdx.y;
     if(tx > 0 && tx < width - 1 && ty > 0 && ty < width - 1) {
    	 float left = grid[getIdx(tx - 1, ty)];
         float right = grid[getIdx(tx + 1, ty)];
         float up = grid[getIdx(tx, ty - 1)];
         float down = grid[getIdx(tx, ty + 1)];
 	     grid[getIdx(tx, ty)] = (left + right + up + down) * 0.25;
	}
}

