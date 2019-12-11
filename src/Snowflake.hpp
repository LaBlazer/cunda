#include "pic_type.hpp"

struct Snowflake {
	Snowflake() {

	}

	Snowflake(int x, int y, float timeScale, float sinScale, int speed, CudaPic pic) :
		originX(x), X(x), Y(y), timeScale(timeScale), sinScale(sinScale), speed(speed), picture(pic) {

	}

	int originX;
	int speed;
	int X;
	int Y;
	float timeScale;
	float sinScale;
	CudaPic picture;
};