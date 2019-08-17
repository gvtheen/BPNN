#include <math.h>
#include <string>
#include <fstream>
#include <algorithm>
#include "BPNNet.h"

namespace BPNN{

BPNNet::BPNNet(MatrixXf* _sampleInput,
               MatrixXf* _sampleOut,
               size_t _hiddenlayerNum,
               const std::vector<size_t> _hiddenNeuronNum,
               float _learnningRate,
               size_t _maxTrainNum,
               float _convergence,
               std::string _learnningAlgorithm,
               std::vector<std::string> _activateFunction
               )
{
    if(_sampleInput->cols() != _sampleOut->cols()){
      ;// treatment under the ERROR
    }
    this->m_sampleNum=_sampleInput->cols();
    this->m_pSampleInput=_sampleInput;
    this->m_pSampleOut=_sampleOut;

    this->m_InputNeuronNum =  m_pSampleInput->rows();
    this->m_OutputNeuronNum = m_pSampleOut->rows();
    this->hidden_layer_num = _hiddenlayerNum;

    if(_hiddenlayerNum!= _hiddenNeuronNum.size()){
      ;// treatment under the ERROR
    }
    this->hidden_neuron_size.assign(_hiddenNeuronNum.begin(),_hiddenNeuronNum.end());

    alpha=_learnningRate;
    lambda=0.0;
    convergence=_convergence;
    m_costValue=1.0;
    trains_num= _maxTrainNum;

    std::vector<std::string> algorithm_label={"traingd","traingdm","traingdx","traincgf","trainrp"};
    if(std::find(algorithm_label.begin(),algorithm_label.end(),_learnningAlgorithm) != algorithm_label.end())
       this->algorithm = this->command2int(_learnningAlgorithm);
    else
       this->algorithm = this->command2int("traincgf");

    std::vector<std::string> func_label={"sigmod","tanh","exp","purelin"};
    for(size_t i=0;i<_activateFunction.size();i++){
       if(std::find(func_label.begin(),func_label.end(),_activateFunction[i]) != func_label.end())
          this->activate_function.push_back(this->command2int(_activateFunction[i]));
       else
          this->activate_function.push_back(this->command2int("sigmod"));

       if(this->activate_function.size()==this->hidden_layer_num+1)
          break;
    }
    for(size_t i=this->activate_function.size();i<this->hidden_layer_num+1; i++)
       this->activate_function.push_back(this->command2int("sigmod"));
}

BPNNet::BPNNet(BPNNet& _oldBPNNet)
{
    this->alpha=_oldBPNNet.alpha;
    this->convergence=_oldBPNNet.convergence;
    this->lambda=_oldBPNNet.lambda;
    this->momentum=_oldBPNNet.momentum;
    this->trains_num=_oldBPNNet.trains_num;
    this->algorithm=_oldBPNNet.algorithm;
    this->activate_function.assign(_oldBPNNet.activate_function.begin(),_oldBPNNet.activate_function.end());

    this->hidden_layer_num=_oldBPNNet.hidden_layer_num;
    this->hidden_neuron_size.assign(_oldBPNNet.hidden_neuron_size.begin(),_oldBPNNet.hidden_neuron_size.end());

    this->m_pInputLayer= new InputLayer();
    m_pInputLayer->value=_oldBPNNet.m_pInputLayer->value;

    this->m_pOutputLayer = new OutputLayer();
    m_pOutputLayer->deta=_oldBPNNet.m_pOutputLayer->deta;
    m_pOutputLayer->dw=_oldBPNNet.m_pOutputLayer->dw;
    m_pOutputLayer->in=_oldBPNNet.m_pOutputLayer->in;
    m_pOutputLayer->out=_oldBPNNet.m_pOutputLayer->out;
    m_pOutputLayer->w=_oldBPNNet.m_pOutputLayer->w;
    m_pOutputLayer->sampleValue=_oldBPNNet.m_pOutputLayer->sampleValue;
    m_pOutputLayer->fun=this->dipatchActiviteFunc(activate_function[activate_function.size()-1]);;
    m_pOutputLayer->dfun=this->dipatchDeriveActiveFunc(activate_function[activate_function.size()-1]);

    for(size_t i=0;i<hidden_layer_num;i++){
        HiddenLayer *temp_Layer= new HiddenLayer();

        temp_Layer->in=_oldBPNNet.m_HiddenLayer[i]->in;
        temp_Layer->out=_oldBPNNet.m_HiddenLayer[i]->out;
        temp_Layer->b=_oldBPNNet.m_HiddenLayer[i]->b;
        temp_Layer->db=_oldBPNNet.m_HiddenLayer[i]->db;
        temp_Layer->w=_oldBPNNet.m_HiddenLayer[i]->w;
        temp_Layer->dw=_oldBPNNet.m_HiddenLayer[i]->dw;
        temp_Layer->pre_dw=_oldBPNNet.m_HiddenLayer[i]->pre_dw;
        temp_Layer->fun=this->dipatchActiviteFunc(activate_function[i]);
        temp_Layer->dfun=this->dipatchDeriveActiveFunc(activate_function[i]);
        this->m_HiddenLayer.push_back(temp_Layer);
    }
}

BPNNet* BPNNet::clone()
{
    return new BPNNet(*this);
}

BPNNet::~BPNNet()
{
    delete this->m_pInputLayer;
    delete this->m_pOutputLayer;
    delete this->m_pSampleInput;
    delete this->m_pSampleOut;
    for(size_t i=0;i<this->hidden_layer_num;i++)
        delete this->m_HiddenLayer[i];
    this->activate_function.clear();
}

void BPNNet::init()
{
    this->m_pInputLayer= new InputLayer();

    for(size_t i=0;i<this->hidden_layer_num;i++){
        HiddenLayer *temp_Layer= new HiddenLayer();

        temp_Layer->in=MatrixXf::Zero(hidden_neuron_size[i],1);
        temp_Layer->out=temp_Layer->in;
        temp_Layer->b=MatrixXf::Zero(hidden_neuron_size[i],1);
        temp_Layer->db=MatrixXf::Zero(hidden_neuron_size[i],1);
        if(i==0){
          temp_Layer->w=MatrixXf::Random(m_InputNeuronNum,hidden_neuron_size[i]);
          temp_Layer->dw=temp_Layer->w;
          temp_Layer->pre_dw=MatrixXf::Zero(m_InputNeuronNum,hidden_neuron_size[i]);
          temp_Layer->pre_db=temp_Layer->pre_dw;
        }else{
          temp_Layer->w=MatrixXf::Random(hidden_neuron_size[i-1],hidden_neuron_size[i]);
          temp_Layer->dw=temp_Layer->w;
          temp_Layer->pre_dw=MatrixXf::Zero(hidden_neuron_size[i-1],hidden_neuron_size[i]);
          temp_Layer->pre_db=temp_Layer->pre_dw;
        }
        temp_Layer->fun=this->dipatchActiviteFunc(activate_function[i]);
        temp_Layer->dfun=this->dipatchDeriveActiveFunc(activate_function[i]);

        this->m_HiddenLayer.push_back(temp_Layer);
    }

    this->m_pOutputLayer = new OutputLayer();
    m_pOutputLayer->w = MatrixXf::Random(hidden_neuron_size[hidden_layer_num-1],m_OutputNeuronNum);
    m_pOutputLayer->dw = m_pOutputLayer->w;
    m_pOutputLayer->pre_dw=MatrixXf::Zero(hidden_neuron_size[hidden_layer_num-1],m_OutputNeuronNum);
    m_pOutputLayer->in = MatrixXf::Zero(m_OutputNeuronNum,1);
    m_pOutputLayer->out = MatrixXf::Zero(m_OutputNeuronNum,1);
    m_pOutputLayer->deta= MatrixXf::Zero(m_OutputNeuronNum,1);
    m_pOutputLayer->fun=this->dipatchActiviteFunc(activate_function[activate_function.size()-1]);;
    m_pOutputLayer->dfun=this->dipatchDeriveActiveFunc(activate_function[activate_function.size()-1]);
}

void BPNNet::forwardpropagation()
{
   for(size_t i=0;i<this->hidden_layer_num;i++){
      if(i==0)
         //( (1*n) * (n*m) )^T = m*1
         m_HiddenLayer[i]->in = (m_pInputLayer->value.transpose()*m_HiddenLayer[i]->w).transpose() + m_HiddenLayer[i]->b;
      else
         m_HiddenLayer[i]->in = (m_HiddenLayer[i-1]->out.transpose()*m_HiddenLayer[i]->w).transpose() + m_HiddenLayer[i]->b;

      for(size_t i=0;i<m_HiddenLayer[i]->in.rows();i++)
            m_HiddenLayer[i]->out(i,0) = (this->*m_HiddenLayer[i]->fun)(m_HiddenLayer[i]->in(i,0));
   }

   m_pOutputLayer->in= (m_HiddenLayer[hidden_layer_num-1]->out.transpose()*m_pOutputLayer->w).transpose();
   for(size_t i=0;i<m_pOutputLayer->in.rows();i++)
      m_pOutputLayer->out(i,0) = (this->*m_pOutputLayer->fun)(m_pOutputLayer->in(i,0));
}
MatrixXf BPNNet::PointMultiply(const MatrixXf& _x,const MatrixXf& _y)
{
   if(_x.rows()!= _y.rows()){

   }
   MatrixXf result(_x.rows(),1);
   for(size_t i=0;i<_x.rows();i++)
      result(i,1)=_x(i,1)*_y(i,1);
   return result;
}

void BPNNet::BPNNet::backpropagation()
{
    MatrixXf diffValue = m_pOutputLayer->out - m_pOutputLayer->sampleValue;
    this->m_costValue = (diffValue.transpose()*diffValue)(0,0);

    MatrixXf m(m_pOutputLayer->in.rows(),1);
    for(size_t i=0;i<m_pOutputLayer->in.rows();i++)
        m(i,0)=(this->*m_pOutputLayer->dfun)(m_pOutputLayer->in(i,0));

    m_pOutputLayer->deta= PointMultiply(diffValue,m);
    MatrixXf m2;
    for(size_t i= hidden_layer_num-1;i>=0;i--){
       if(i==hidden_layer_num-1)
          m=m_pOutputLayer->w*m_pOutputLayer->deta;
       else
          m=m_HiddenLayer[i+1]->w*m_HiddenLayer[i+1]->deta;

       m2=MatrixXf::Zero(m_HiddenLayer[i]->in.rows(),1);
       for(size_t j=0;j<m_HiddenLayer[i]->in.rows();j++)
           m2(j,0)=(this->*m_HiddenLayer[i]->dfun)(m_HiddenLayer[i]->in(j,0));

       m_HiddenLayer[i]->deta = PointMultiply(m,m2);
    }

    for(size_t i=0;i<this->hidden_layer_num;i++){
        if(i==0)
           m_HiddenLayer[i]->dw = m_pInputLayer->value*(m_HiddenLayer[i]->deta.transpose())/m_sampleNum;
        else
           m_HiddenLayer[i]->dw = m_HiddenLayer[i-1]->out*(m_HiddenLayer[i]->deta.transpose())/m_sampleNum;
        m_HiddenLayer[i]->db=m_HiddenLayer[i]->deta;
    }
    m_pOutputLayer->dw=m_HiddenLayer[hidden_layer_num-1]->out*(m_pOutputLayer->deta.transpose())/m_sampleNum;
}

void BPNNet::update_weight()
{
    switch(this->algorithm){
       case GD_algorithm:
            traingd_study();
           break;
       case GDM_algorithm:
            traingdm_study();
            break;
       case GDX_algorithm:
            traingdx_study();
            break;
       case CGF_algorithm:
            traincgf_study();
            break;
       case RP_algorithm:
            trainrp_study();
            break;
       default:
            traingd_study();
            break;
    }
}

void BPNNet::train()
{
   train_epoch=0;
   m_Error = MatrixXf::Zero(m_sampleNum,trains_num);
   while(1){

         for(size_t i=0;i<this->m_sampleNum;i++){
            //read sample
            m_pInputLayer->value= m_pSampleInput->col(i);
            m_pOutputLayer->sampleValue=m_pSampleOut->col(i);
            // forward computation
            this->forwardpropagation();
            // back propagation
            this->backpropagation();
            this->update_weight();
            m_Error(i,train_epoch)= m_costValue;
         }
         this->output();

         if(m_Error.col(train_epoch).sum()/m_sampleNum < this->convergence)
            break;

         train_epoch++;
         if(train_epoch > this->trains_num){
            break;
            // other output
         }
   }
}

MatrixXf BPNNet::predict(const MatrixXf *_testSample)
{
   MatrixXf singleTest(_testSample->rows(),1);

   size_t totalSampleSize = _testSample->cols();

   MatrixXf result(this->m_OutputNeuronNum,totalSampleSize);

   BPNNet *bestBPnet=nullptr;
   for(size_t i=0;i<totalSampleSize;i++){
        bestBPnet  = this->clone();
        singleTest = _testSample->col(i);
        for(size_t i=0;i<bestBPnet->hidden_layer_num;i++){
           if(i==0)
              bestBPnet->m_HiddenLayer[i]->in = (singleTest.transpose()*           \
                                                bestBPnet->m_HiddenLayer[i]->w).transpose() + bestBPnet->m_HiddenLayer[i]->b;
           else
              bestBPnet->m_HiddenLayer[i]->in = (bestBPnet->m_HiddenLayer[i-1]->out.transpose()*      \
                                                bestBPnet->m_HiddenLayer[i]->w).transpose() + bestBPnet->m_HiddenLayer[i]->b;

           for(size_t i=0;i<bestBPnet->m_HiddenLayer[i]->in.rows();i++)
              bestBPnet->m_HiddenLayer[i]->out(i,0) =(bestBPnet->*m_HiddenLayer[i]->fun)(bestBPnet->m_HiddenLayer[i]->in(i,0));
        }
        bestBPnet->m_pOutputLayer->in = (bestBPnet->m_HiddenLayer[bestBPnet->hidden_layer_num - 1]->out.transpose()* \
                                         bestBPnet->m_pOutputLayer->w).transpose() + \
                                         bestBPnet->m_HiddenLayer[bestBPnet->hidden_layer_num - 1]->b;
        for(size_t i=0;i<bestBPnet->m_pOutputLayer->in.rows();i++)
            bestBPnet->m_pOutputLayer->out(i,0) =(bestBPnet->*m_pOutputLayer->fun)(bestBPnet->m_pOutputLayer->in(i,0));
        result.col(i) = bestBPnet->m_pOutputLayer->out;
   }
   return result;
}

void BPNNet::traingd_study()
{
   for(size_t i=0;i<this->hidden_layer_num;i++){
        m_HiddenLayer[i]->w = m_HiddenLayer[i]->w - alpha*m_HiddenLayer[i]->dw;
        m_HiddenLayer[i]->b = m_HiddenLayer[i]->b - alpha*m_HiddenLayer[i]->db;
   }
   m_pOutputLayer->w =m_pOutputLayer->w - alpha*m_pOutputLayer->dw;
}
void BPNNet::traingdm_study()
{
   MatrixXf tmp;
   for(size_t i=0;i<this->hidden_layer_num;i++){
     tmp = m_HiddenLayer[i]->dw;
     m_HiddenLayer[i]->w = m_HiddenLayer[i]->w - alpha*m_HiddenLayer[i]->dw + momentum*m_HiddenLayer[i]->pre_dw;
     m_HiddenLayer[i]->pre_dw = tmp;
     tmp = m_HiddenLayer[i]->db;
     m_HiddenLayer[i]->b = m_HiddenLayer[i]->b - alpha*m_HiddenLayer[i]->db + momentum*m_HiddenLayer[i]->pre_db;
     m_HiddenLayer[i]->pre_db = tmp;
   }
   tmp = m_pOutputLayer->dw;
   m_pOutputLayer->w =m_pOutputLayer->w - alpha*m_pOutputLayer->dw + momentum*m_pOutputLayer->pre_dw;
   m_pOutputLayer->pre_dw = tmp;
}

void BPNNet::traingdx_study()
{
   if(this->alpha_vector.size()==0 || this->train_epoch == this->alpha_vector.size())
      this->alpha_vector.push_back(this->alpha);

   float modified_alpha=0;
   if(this->train_epoch==0)
            ;
   else if(m_Error.col(train_epoch).sum()/m_sampleNum < m_Error.col(train_epoch-1).sum()/m_sampleNum){
     modified_alpha = alpha_vector[alpha_vector.size()-1];
     alpha_vector.push_back(modified_alpha*1.05);
   }else if(m_Error.col(train_epoch).sum()/m_sampleNum > m_Error.col(train_epoch-1).sum()/m_sampleNum){
     modified_alpha = alpha_vector[alpha_vector.size()-1];
     alpha_vector.push_back(modified_alpha*0.70);
   }
    modified_alpha=alpha_vector[alpha_vector.size()-1];
    for(size_t i=0;i<this->hidden_layer_num;i++){
        m_HiddenLayer[i]->w = m_HiddenLayer[i]->w - modified_alpha*m_HiddenLayer[i]->dw;
        m_HiddenLayer[i]->b = m_HiddenLayer[i]->b - modified_alpha*m_HiddenLayer[i]->db;
    }
   m_pOutputLayer->w =m_pOutputLayer->w - modified_alpha*m_pOutputLayer->dw;
}
void BPNNet::trainrp_study()
{

}

void BPNNet::traincgf_study()
{

}

bool BPNNet::load(const std::string & file_name)
{
    std::ifstream in(file_name.c_str(),std::ios::binary);
    long int code;
    in.read((char*)(&code),sizeof(long int));
    if(code != 19080423)
        return false;
    in.read((char*)(&hidden_layer_num),sizeof(size_t));
    for(size_t i=0;i<this->hidden_layer_num;i++)
        in.read((char*)(&hidden_neuron_size[i]),sizeof(size_t));
    in.read((char*)(&m_InputNeuronNum),sizeof(size_t));
    in.read((char*)(&m_OutputNeuronNum),sizeof(size_t));
    // write optimization algorithm
    in.read((char*)(&algorithm),sizeof(int));
    // write activation function
    for(size_t i=0;i<this->activate_function.size();i++)
        in.read((char*)(&activate_function[i]),sizeof(int));
    // write study rate and momentum
    in.read((char*)(&alpha),sizeof(float));
    in.read((char*)(&momentum),sizeof(float));
    // write the weight and bias of hidden layer
    for(size_t i=0;i<this->hidden_layer_num;i++){
        for(size_t j=0;j<this->m_HiddenLayer[i]->w.rows();j++)
           for(size_t k=0;k<this->m_HiddenLayer[i]->w.cols();k++){
             in.read((char*)&(m_HiddenLayer[i]->w(j,k)),sizeof(float));
             in.read((char*)&(m_HiddenLayer[i]->b(j,k)),sizeof(float));
           }
        m_HiddenLayer[i]->fun=this->dipatchActiviteFunc(activate_function[i]);
        m_HiddenLayer[i]->dfun=this->dipatchDeriveActiveFunc(activate_function[i]);
    }
   for(size_t j=0;j<this->m_pOutputLayer->w.rows();j++)
       for(size_t k=0;k<this->m_pOutputLayer->w.cols();k++)
             in.read((char*)&(m_pOutputLayer->w(j,k)),sizeof(float));
   m_pOutputLayer->fun=this->dipatchActiviteFunc(activate_function[activate_function.size()-1]);;
   m_pOutputLayer->dfun=this->dipatchDeriveActiveFunc(activate_function[activate_function.size()-1]);
   // close the file pointer
   in.close();
}

void BPNNet::save(const std::string & file_name)
{
    std::ofstream out(file_name.c_str(),std::ios::binary);
    //write the handshake protocol
    out.write((char*)(19080423),sizeof(long int));
    //write information of NN framework
    out.write((char*)(&hidden_layer_num),sizeof(size_t));
    for(size_t i=0;i<this->hidden_layer_num;i++)
        out.write((char*)(&hidden_neuron_size[i]),sizeof(size_t));
    out.write((char*)(&m_InputNeuronNum),sizeof(size_t));
    out.write((char*)(&m_OutputNeuronNum),sizeof(size_t));
    // write optimization algorithm
    out.write((char*)(&algorithm),sizeof(int));
    // write activation function
    for(size_t i=0;i<this->activate_function.size();i++)
        out.write((char*)(&activate_function[i]),sizeof(int));
    // write study rate and momentum
    out.write((char*)(&alpha),sizeof(float));
    out.write((char*)(&momentum),sizeof(float));
    // write the weight and bias of hidden layer
    for(size_t i=0;i<this->hidden_layer_num;i++)
        for(size_t j=0;j<this->m_HiddenLayer[i]->w.rows();j++)
           for(size_t k=0;k<this->m_HiddenLayer[i]->w.cols();k++){
             out.write((char*)&(m_HiddenLayer[i]->w(j,k)),sizeof(float));
             out.write((char*)&(m_HiddenLayer[i]->b(j,k)),sizeof(float));
           }

   // write the weight and bias of output layer
   for(size_t j=0;j<this->m_pOutputLayer->w.rows();j++)
       for(size_t k=0;k<this->m_pOutputLayer->w.cols();k++)
             out.write((char*)&(m_pOutputLayer->w(j,k)),sizeof(float));

   out.close();
}

float BPNNet::fsigmod(const float &x)
{
     return 1.0 / (1.0 + exp(-x));
}

float BPNNet::dsigmod(const float &x)
{
     float y=fsigmod(x);
     return y*(1-y);
}

float BPNNet::ftanh(const float& x)
{
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

float BPNNet::dtanh(const float& x)
{
    float y=ftanh(x);
    return 1-y*y;
}

float BPNNet::fexp(const float& x)
{
    return exp(-0.5*x*x);
}

float BPNNet::dexp(const float& x)
{
   float y=fexp(x);
   return -y*exp(-0.5*y*y);
}

float BPNNet::fpurelin(const float &x)
{
    return x*0.1;
}

float BPNNet::dpurelin(const float &x)
{
    return 0.1;
}

bool BPNNet::setLearnRate(const float& x)
{
    this->alpha=x;
    return true;
}

bool BPNNet::setMomentum(const float& x)
{
    this->momentum=x;
    return true;
}

BPNNet::FUNc BPNNet::dipatchActiviteFunc(const int& _ix)
{
   switch (_ix){
     case SIGMOD_function:
       return &BPNNet::fsigmod;
       break;
     case TANH_function:
       return &BPNNet::ftanh;
       break;
     case EXP_function:
       return &BPNNet::fexp;
       break;
     case PURELIN_function:
       return &BPNNet::fpurelin;
       break;
     default:
       return &BPNNet::fsigmod;
       break;
   }
}

BPNNet::FUNc BPNNet::dipatchDeriveActiveFunc(const int& _ix)
{
   switch (_ix){
     case SIGMOD_function:
       return &BPNNet::dsigmod;
       break;
     case TANH_function:
       return &BPNNet::dtanh;
       break;
     case EXP_function:
       return &BPNNet::dexp;
       break;
     case PURELIN_function:
       return &BPNNet::dpurelin;
       break;
     default:
       return &BPNNet::dsigmod;
       break;
   }
}

int BPNNet::command2int(const std::string& _x)
{
    if(_x =="traingd")
         return GD_algorithm;
    else if(_x=="traingdm")
         return GDM_algorithm;
    else if(_x=="traingdx")
         return GDX_algorithm;
    else if(_x=="traincgf")
         return CGF_algorithm ;
    else if(_x=="trainrp")
         return RP_algorithm;
    else if(_x=="sigmod")
         return SIGMOD_function;
    else if(_x=="tanh")
         return TANH_function;
    else if(_x=="exp")
         return EXP_function;
    else if(_x=="purelin")
         return PURELIN_function;
    else
         return -1;
}


}
