#include<iostream>
#include <Eigen/Dense>

using namespace std;

double relu(double x)
{
	return max(0.0, x);
}

double Drelu(double x)
{
	if (x > 0.0)
		return 1.0;
	else if (x < 0.0)
		return 0.0;
	else
		return 0.5;
}

template<typename T, int backunits_len, int units_len>// If it's input layer, there is no backunit so we will set backunits_len=units_len.
class DenseLayer
{
public:
	DenseLayer(double _learning_rate=0.03,bool _is_input_layer=false );
	void Initializer();
	Eigen::Matrix<double, 1, Eigen::Dynamic> ForwardPropagation(Eigen::Matrix<double, 1, backunits_len> _x_data);
	Eigen::Matrix<double, units_len, units_len> cal_gradient();
	Eigen::Matrix<double, 1, backunits_len> BackwardPropagation(Eigen::Matrix<double, 1, units_len>  gradient);
private:
	bool is_input_layer;
	double learning_rate;
	Eigen::Matrix<double, 1, units_len> output;
	Eigen::Matrix<double, 1, units_len> wx_plus_b;
	Eigen::Matrix<double,1, units_len> bias;
	Eigen::Matrix<double, backunits_len, units_len> weight;
	Eigen::Matrix<double, 1, backunits_len> x_data;
	Eigen::Matrix<double, 1, backunits_len> gradient_to_prop;
	Eigen::Matrix<double, backunits_len, units_len> gradient_weight;
	Eigen::Matrix<double, 1, units_len> gradient_b;
};

template<typename T, int backunits_len, int units_len>
DenseLayer<T, backunits_len, units_len>::DenseLayer(double _learning_rate = 0.03, bool _is_input_layer = false)
{
	cout << "Construct a layer "<< backunits_len<<" to "<< units_len <<"!"<< endl;
	is_input_layer = _is_input_layer;
	learning_rate = _learning_rate;
}

template<typename T, int backunits_len, int units_len>
void DenseLayer<T, backunits_len, units_len>::Initializer()
{
	cout << "Initialize a layer " << backunits_len << " to " << units_len << "!" << endl;
	weight = Eigen::Matrix<double, backunits_len, units_len>::Random();
	bias = Eigen::Matrix<double, 1, units_len>::Random();
}

template<typename T, int backunits_len, int units_len>
Eigen::Matrix<double, 1, Eigen::Dynamic> DenseLayer<T, backunits_len, units_len>::ForwardPropagation(Eigen::Matrix<double, 1, backunits_len> _x_data)
{
	x_data = _x_data;
	if (is_input_layer==true)
	{
		return x_data;
	}
	else
	{
		wx_plus_b = x_data*weight - bias;
		output = wx_plus_b.unaryExpr([](double x) { return relu(x); });
		return output;
	}
}

template<typename T, int backunits_len, int units_len>
Eigen::Matrix<double, units_len, units_len> DenseLayer<T, backunits_len, units_len>::cal_gradient()
{
	// Calculate a diagnal matrix to represent 1{wx_plus_b[i]>=0}, return a  units_len * units_len matrix.
	return wx_plus_b.unaryExpr([](double x) { return Drelu(x); }).asDiagonal();

}

template<typename T, int backunits_len, int units_len>
Eigen::Matrix<double, 1, backunits_len> DenseLayer<T, backunits_len, units_len>::BackwardPropagation(Eigen::Matrix<double, 1, units_len> gradient)
{
	//partial loss/ partial wij= 1{wx_plus_b[i]>=0} * xdatai * gradientj
	Eigen::Matrix<double, units_len, units_len> gradient_activation_weight = cal_gradient();
	double gradient_activation_b = -1.0;


	gradient_weight = x_data.transpose()*gradient*gradient_activation_weight; //(backunits,1)*(1,units)*(units,units)
	gradient_b = gradient*gradient_activation_b; //(backunits,1)*(1,units)*(units,units)

	weight = weight - learning_rate*gradient_weight;
	bias = bias - learning_rate*gradient_b;

	gradient_to_prop = gradient*(weight*gradient_activation_weight).transpose(); //(1,units)*[(backunits,units)*(units,units)].T

	return gradient_to_prop;

}


class BPNN
{

};

int main()
{
	cout << "Hello world!" << endl;
	DenseLayer<double,3,4> D(0.03,false);
	D.Initializer();
	Eigen::Matrix<double, 1, 3> x_data;
	x_data << 1, 2, 3;
	cout<<D.ForwardPropagation(x_data)<<endl;
	cout << D.cal_gradient() << endl;

	Eigen::Matrix<double, 1, 4>  gradient;
	gradient << 1, 2, 3, 4;
	cout << D.BackwardPropagation(gradient) << endl;
	cout << endl;

	Eigen::Matrix3f m;
	m << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	std::cout << m << std::endl;
	system("pause");
	return 0;
}