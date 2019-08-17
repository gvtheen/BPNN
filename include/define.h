#ifndef DEFINE_H_INCLUDED
#define DEFINE_H_INCLUDED
#include<vector>

using namespace std;

namespace BPNN{

struct Neuron
{
	std::vector<double> weight;
	std::vector<double> update_w;
	double input;
	double output = 0;
	double bias;
};
typedef std::vector<Neuron> Layer;

}
#endif // DEFINE_H_INCLUDED
