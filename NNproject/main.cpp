#include<iostream>
#include <Eigen/Dense>
#include<vector>

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

double sigmoid(double x)
{
	return 1/(1+exp(-x));
}

double Dsigmoid(double x)
{
	return sigmoid(x)*(1 - sigmoid(x));
}

class DenseLayer
{
public:
	DenseLayer(int _backunits_len, int _units_len, double _learning_rate,bool _is_input_layer);
	void Initializer();
	Eigen::Matrix<double, 1, Eigen::Dynamic> ForwardPropagation(Eigen::Matrix<double, 1, Eigen::Dynamic> _x_data);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cal_gradient();
	Eigen::Matrix<double, 1, Eigen::Dynamic> BackwardPropagation(Eigen::Matrix<double, 1, Eigen::Dynamic>  gradient);
	int getbackunits() { return backunits_len; };
	int getunits() { return units_len; };
	void setinputlayer() { is_input_layer = true; };
private:
	int backunits_len; int units_len;
	bool is_input_layer;
	double learning_rate;
	Eigen::Matrix<double, 1, Eigen::Dynamic> output;
	Eigen::Matrix<double, 1, Eigen::Dynamic> wx_plus_b;
	Eigen::Matrix<double,1, Eigen::Dynamic> bias;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weight;
	Eigen::Matrix<double, 1, Eigen::Dynamic> x_data;
	Eigen::Matrix<double, 1, Eigen::Dynamic> gradient_to_prop;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_weight;
	Eigen::Matrix<double, 1, Eigen::Dynamic> gradient_b;
};


DenseLayer::DenseLayer(int _backunits_len, int _units_len, double _learning_rate = 0.03, bool _is_input_layer = false)
{
	is_input_layer = _is_input_layer;
	learning_rate = _learning_rate;
	backunits_len = _backunits_len;
	units_len = _units_len;
	cout << "Construct a layer " << backunits_len << " to " << units_len << "!" << endl;
}


void DenseLayer::Initializer()
{
	weight = Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic>::Random(backunits_len, units_len);
	bias = Eigen::Matrix<double, 1, Eigen::Dynamic>::Random(1, units_len);
	cout << "Initialize a layer " << backunits_len << " to " << units_len << "!" << endl;
}

Eigen::Matrix<double, 1, Eigen::Dynamic> DenseLayer::ForwardPropagation(Eigen::Matrix<double, 1, Eigen::Dynamic> _x_data)
{
	x_data = _x_data;
	if (is_input_layer==true)
	{
		return x_data;
	}
	else
	{
		wx_plus_b = x_data*weight - bias;
		output = wx_plus_b.unaryExpr([](double x) { return sigmoid(x); });
		return output;
	}
}


Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DenseLayer::cal_gradient()
{
	// Calculate a diagnal matrix to represent 1{wx_plus_b[i]>=0}, return a  units_len * units_len matrix.
	return wx_plus_b.unaryExpr([](double x) { return Dsigmoid(x); }).asDiagonal();

}


Eigen::Matrix<double, 1, Eigen::Dynamic> DenseLayer::BackwardPropagation(Eigen::Matrix<double, 1, Eigen::Dynamic> gradient)
{
	//partial loss/ partial wij= 1{wx_plus_b[i]>=0} * xdatai * gradientj
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient_activation_weight = cal_gradient();
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
public:
	BPNN();
	void AddLayer(DenseLayer *layer);
	void BuildLayer();
	void Summary();
	double Train(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>xdata, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>ydata, int _train_round, double _accuracy);
	double cal_loss(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>ydata, Eigen::Matrix<double, Eigen::Dynamic, 1>ydata_);
private:
	vector<DenseLayer*> layers;
	vector<double> train_mse;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>loss_gradient;
	int train_round;
	double accuracy;
};

BPNN::BPNN()
{

}

void BPNN::AddLayer(DenseLayer *layer)
{
	layers.push_back(layer);
}

void BPNN::BuildLayer()
{
	for (int i=0;i<layers.size();i++)
	{
		if (i == 0)
			layers[i]->setinputlayer();
		layers[i]->Initializer();
	}
}

void BPNN::Summary()
{
	for (int i = 0; i<layers.size(); i++)
	{
		cout<<"-------------"<<i<<"th layer-------------"<<endl;
		cout << "weight shape = " << layers[i]->getbackunits() << "*"<<layers[i]->getunits() << endl;
	}
}

double BPNN::Train(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>xdata, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>ydata, int _train_round, double _accuracy )
{
	train_round = _train_round;
	accuracy = _accuracy;
	cout << _accuracy << endl;

	int n = xdata.rows();
	double loss = 0;
	double all_loss = 0;

	if (n != ydata.rows())
	{
		cout << "Bad input data!" << endl;
		return 0;
	}

	for (int i = 0; i < train_round; i++)
	{
		all_loss = 0;
		for (int j = 0; j < n; j++)
		{
			Eigen::Matrix<double, 1, Eigen::Dynamic>_xdata = xdata.row(j);
			Eigen::Matrix<double, 1, Eigen::Dynamic>_ydata = ydata.row(j);

			for (auto layer : layers)
			{
				_xdata = layer->ForwardPropagation(_xdata);
			}

			loss_gradient = 2.0 * (_xdata - _ydata);
			loss = loss_gradient.unaryExpr([](double x) { return x*x / 4.0; }).sum();

			all_loss += loss;

			for (int k = 0; k < layers.size() - 1; k++)
			{
				loss_gradient = layers[layers.size() - 1 - k]->BackwardPropagation(loss_gradient);
			}
		}
		double mse = all_loss / n;
		train_mse.push_back(mse);
		/*if (abs(train_mse[train_mse.size() - 2] - train_mse[train_mse.size() - 1]) < accuracy)
		{
			cout << "Satisfy accuracy!" << endl;
			return mse;
		}*/
		cout << mse << endl;
		if (mse < accuracy)
		{
			cout << "Satisfy accuracy!" << endl;
			return mse;
		}
	}

	return 0;
}


int main()
{
	cout << "Hello world!" << endl;
	DenseLayer D(3,5,0.03,false);
	D.Initializer();
	Eigen::Matrix<double, 1, 3> x_data;
	x_data << 1, 2, 3;
	cout<<D.ForwardPropagation(x_data)<<endl;
	cout << D.cal_gradient() << endl;

	Eigen::Matrix<double, 1, 5>  gradient;
	gradient << 1, 2, 3, 4, 5;
	cout << D.BackwardPropagation(gradient) << endl;
	cout << endl;

	//test the training process

	double learning_rate = 0.3;
	BPNN modelnew;
	DenseLayer* layer1 = new DenseLayer(10, 10, learning_rate, true);
	DenseLayer* layer2 = new DenseLayer(10, 20, learning_rate, false);
	DenseLayer* layer3 = new DenseLayer(20, 30, learning_rate, false);
	DenseLayer* layer4 = new DenseLayer(30, 2, learning_rate, false);

	modelnew.AddLayer(layer1);
	modelnew.AddLayer(layer2);
	modelnew.AddLayer(layer3);
	modelnew.AddLayer(layer4);
	modelnew.BuildLayer();
	modelnew.Summary();

	Eigen::Matrix<double, 10, 10> x;
	x << -0.42341286, 0.21779802, -0.54369312, 2.04964989, 1.00671986,
		0.72770789, 0.2580108, 0.74788435, 1.45180192, 0.86803638,
		-1.43974545, -1.20253251, -1.24224465, 0.24809309, -0.93821806,
		1.29316884, -0.50198725, -0.63714213, 0.12479802, 0.91007394,
		-0.78658784, -1.12794307, -0.77812005, 1.29574899, 0.16750844,
		-0.70761621, 1.51739084, -1.19870489, -1.53029875, -0.9038248,
		-0.9756778, 0.66175796, 0.26833978, 1.75458108, 0.15402258,
		-0.42806397, -0.63166847, 0.19717951, -1.97259133, 0.23806793,
		0.83755467, -0.37247964, -0.06758306, 0.22669441, -0.1273009,
		1.47156685, 0.30417944, 1.66046617, 1.0805952, 1.02822416,
		-1.30650562, 0.66428356, -1.51496519, 0.30665193, -0.95840903,
		0.69387956, 0.54239419, -0.13788214, 1.14797255, -1.18778428,
		0.92176127, -0.37185503, -0.51249125, -1.52096541, 0.392217,
		-1.26853408, -0.23724684, 0.72507058, 0.0810218, 1.20581851,
		0.55981882, -1.77590695, -1.12788518, -0.02926117, 0.31905083,
		1.11389359, -0.56559586, 0.10578212, -1.30172802, 1.84858769,
		-0.2738502, -1.44412151, 0.7872747, -0.10611829, 1.06023464,
		-0.12080409, -1.38991104, -0.51387999, 0.9472472, 0.28645597,
		0.10045478, 0.2806141, 0.12326028, 0.5001843, 0.22650803,
		-0.66142985, -0.50764307, 1.35874742, -0.54401188, 1.11425037;
	Eigen::Matrix<double, 10, 2> y;
	y << 0.8, 0.4, 0.4, 0.3, 0.34, 0.45, 0.67, 0.32,
		0.88, 0.67, 0.78, 0.77, 0.55, 0.66, 0.55, 0.43, 0.54, 0.1,
		0.1, 0.5;
	modelnew.Train(x, y, 1000, 0.01);

	delete layer1;
	delete layer2;
	delete layer3; 
	delete layer4;

	system("pause");
	return 0;
}