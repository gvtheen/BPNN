#ifndef CBPNN_H
#define CBPNN_H
#include<iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "define.h"
using namespace Eigen;
namespace BPNN{

class BPNNet
{
    #define    GD_algorithm             0          // Gradient descent algorithm
    #define    GDM_algorithm            1          // Gradient descent algorithm with momentum
    #define    GDX_algorithm            2          // Gradient descent algorithm with self-study
    #define    CGF_algorithm            3          // conjugate gradient
    #define    RP_algorithm             4          //

    #define    SIGMOD_function          10
    #define    TANH_function            11
    #define    EXP_function             12
    #define    PURELIN_function         13

    typedef float (BPNNet::*FUNc)(const float &x);

    typedef struct _input_layer{
       MatrixXf value;
    }InputLayer;

    typedef struct _hidden_layer{
       MatrixXf in;
       MatrixXf out;
       MatrixXf w,dw,deta,pre_dw;
       MatrixXf b,db,pre_db;
           FUNc fun;   //activation function
           FUNc dfun;  //derive of activation function
    }HiddenLayer;

    typedef struct _output_layer{
       MatrixXf in;  // in = sum( out of last hidden layer * w + b )
       MatrixXf out;
       MatrixXf w,dw,deta,pre_dw;
           FUNc fun;   //activation function
           FUNc dfun;  //derive of activation function
       MatrixXf sampleValue;
    }OutputLayer;

    public:
        BPNNet(MatrixXf* sampleInput,
               MatrixXf* sampleOut,
               size_t hiddenlayerNum=1,
               const std::vector<size_t> hiddenNeuronNum={6},
               float _learnningRate=0.9,
               size_t _maxTrainNum=100,
               float _convergence=0.001,
               std::string _learnningAlgorithm="traincg",
               std::vector<std::string> _activateFunction={"sigmod","sigmod"}
               );
        BPNNet(BPNNet&);
        BPNNet* clone();

        virtual ~BPNNet();
        void init();
	    void train();
	    MatrixXf predict(const MatrixXf *_predictSample);

	    bool load(const std::string & file_name="bpnn_optimal_para.nn");
	    void save(const std::string & file_name="bpnn_optimal_para.nn");

	public:   //activation function
       float fsigmod(const float &x);
       float dsigmod(const float &x);

       float ftanh(const float& x);
       float  dtanh(const float& x);

       float fexp(const float& x);
       float dexp(const float& x);

       float fpurelin(const float &x);
       float dpurelin(const float &x);
    private:
       void forwardpropagation();
       void backpropagation();
       void update_weight();
	   void output() const;
	   void traingd_study();
	   void traingdm_study();
       void traingdx_study();
	   void trainrp_study();
	   void traincgf_study();

	   FUNc dipatchActiviteFunc(const int&);
	   FUNc dipatchDeriveActiveFunc(const int&);
	   int  command2int(const std::string&);
       MatrixXf PointMultiply(const MatrixXf&,const MatrixXf&);

    public:
        bool setLearnRate(const float&);
        bool setMomentum(const float&);

    public:
       vector<size_t> hidden_neuron_size;  //
       size_t hidden_layer_num;
       float momentum;
       float alpha;        //learning rate
       float lambda;             //weight decay,
       float convergence;       //convergence value and cost function
       size_t trains_num;

       MatrixXf  *m_pSampleInput, *m_pSampleOut;
       OutputLayer  *m_pOutputLayer;
       std::vector<HiddenLayer*> m_HiddenLayer;
       InputLayer   *m_pInputLayer;
       MatrixXf  m_Error;

       int algorithm;
       std::vector<int> activate_function;
    private:
       size_t m_InputNeuronNum;
       size_t m_OutputNeuronNum;
       float m_costValue;
       size_t m_sampleNum;

       size_t train_epoch;
       std::vector<float> alpha_vector;   //used for traingdx


};


}
#endif // CBPNN_H
