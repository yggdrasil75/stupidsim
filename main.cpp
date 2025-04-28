#include <string>
#include <iostream>
#include <fstream>
#include "icosphere.cpp"


float RADIUS = 637100;
int PLATES = 15;
int SUBDIVISIONS = 3;

int main(int argc, char* argv[]) {
	float radius = RADIUS;
	int subdivisions = SUBDIVISIONS;
	int plates = PLATES;
	std::string outputFile = "out.obj";
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "-radius" && i + 1 < argc) {
			radius = std::stof(argv[i+1]);
		}
		if (arg == "-plates" && i + 1 < argc) {
			plates = std::stoi(argv[i+1]);
		}
		if (arg == "-subdivisions" && i + 1 < argc) {
			subdivisions = std::stoi(argv[i+1]);
		}
	}
	worldState worldsim = icosphere(radius, subdivisions);

	std::ofstream outFile(outputFile);
	if (!outFile.is_open()) {
		return 1;
	}

}