#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include <random>
#include <chrono>
#include "Eigen/Dense"
#define stock_rows 465
#define stock_cols 487

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
	return 1 / (1 + exp(-x));
}

double Dsigmoid(double x)
{
	return sigmoid(x)*(1 - sigmoid(x));
}

double Dnone(double x)
{
	return 1;
}

double x_square_by_4(double x)
{
    return x*x / 4.0; 
}

double x_square(double x)
{
	return x*x; 
}


class DenseLayer
{
public:
	enum Activation { Sigmoid, ReLu, None };
	enum Optimizer {GD, CGD};//Ludi Wang
	DenseLayer(int _backunits_len, int _units_len, double _learning_rate, bool _is_input_layer, Activation t);
	void Initializer();
	Eigen::MatrixXd ForwardPropagation(const Eigen::MatrixXd &_x_data);
	Eigen::MatrixXd cal_gradient();
	Eigen::MatrixXd BackwardPropagation(const Eigen::MatrixXd & gradient);
	int getbackunits() { return backunits_len; };
	int getunits() { return units_len; };
	void setinputlayer() { is_input_layer = true; };
private:
	Optimizer o;
	Activation act_func;
	int backunits_len; int units_len;
	bool is_input_layer;
	double learning_rate;
	Eigen::MatrixXd output;
	Eigen::MatrixXd wx_plus_b;
	Eigen::MatrixXd bias;
	Eigen::MatrixXd weight;
	Eigen::MatrixXd x_data;
	Eigen::MatrixXd gradient_to_prop;
	Eigen::MatrixXd gradient_weight;
	Eigen::MatrixXd gradient_b;
};


DenseLayer::DenseLayer(int _backunits_len, int _units_len, double _learning_rate = 0.03, bool _is_input_layer = false, Activation t = DenseLayer::Sigmoid):
	output(1, _units_len),wx_plus_b(1, _units_len), bias(1, _units_len), weight(_backunits_len, _units_len), x_data(1, _backunits_len), gradient_to_prop(1, _backunits_len),
	gradient_weight(_backunits_len, _units_len), gradient_b(1, _units_len)
{
	is_input_layer = _is_input_layer;
	learning_rate = _learning_rate;
	backunits_len = _backunits_len;
	units_len = _units_len;
	act_func = t;

	cout << "Construct a layer " << backunits_len << " to " << units_len << "!" << endl;
}


void DenseLayer::Initializer()
{
	/*weight = Eigen::MatrixXd::Random(backunits_len, units_len);
	bias = Eigen::MatrixXd::Random(1, units_len);*/

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen;
	std::normal_distribution<double> dis(0, 0.1);
	for (int i = 0; i < backunits_len; i++)
		for (int j = 0; j < units_len; j++)
			weight(i, j) = dis(gen);
	for (int j = 0; j < units_len; j++)
		bias(0, j) = dis(gen);

	cout << "Initialize a layer " << backunits_len << " to " << units_len << "!" << endl;
}

Eigen::MatrixXd DenseLayer::ForwardPropagation(const Eigen::MatrixXd &_x_data)
{
	x_data = _x_data;
	if (is_input_layer == true)
	{
		return x_data;
	}
	else
	{
		wx_plus_b = x_data*weight - bias;
		if (act_func == Activation::Sigmoid)
			output = wx_plus_b.unaryExpr(std::ref(sigmoid));//[](double x) { return sigmoid(x); });
		else if (act_func == Activation::ReLu)
			output = wx_plus_b.unaryExpr(std::ref(relu));//[](double x) { return relu(x); });
		else if (act_func == Activation::None)
			output = wx_plus_b;
		return output;
	}
}


Eigen::MatrixXd DenseLayer::cal_gradient()
{
	// Calculate a diagnal matrix to represent 1{wx_plus_b[i]>=0}, return a  units_len * units_len matrix.
	Eigen::Matrix<double, 1, Eigen::Dynamic> D;
	if (act_func == Activation::Sigmoid)
		D = wx_plus_b.unaryExpr(std::ref(Dsigmoid));//[](double x) { return Dsigmoid(x); });
	else if (act_func == Activation::ReLu)
		D = wx_plus_b.unaryExpr(std::ref(Drelu));//[](double x) { return Drelu(x); });
	else if (act_func == Activation::None)
		D = wx_plus_b.unaryExpr(std::ref(Dnone));//[](double x) { return 1; });
	return D.asDiagonal();

}


Eigen::MatrixXd DenseLayer::BackwardPropagation(const Eigen::MatrixXd &gradient)
{
	//partial loss/ partial wij= 1{wx_plus_b[i]>=0} * xdatai * gradientj
	Eigen::MatrixXd gradient_activation = cal_gradient();

	gradient_weight = x_data.transpose()*gradient*gradient_activation; //(backunits,1)*(1,units)*(units,units)
	gradient_b = -gradient*gradient_activation; //(1,units)*(units,units)

	//if(o==Optimizer::GD)
	//{

	//}

	weight = weight - learning_rate*gradient_weight;
	bias = bias - learning_rate*gradient_b;

	gradient_to_prop = gradient*(weight*gradient_activation).transpose(); //(1,units)*[(backunits,units)*(units,units)].T

	return gradient_to_prop;

}


class BPNN
{
public:
	BPNN();
	~BPNN();
	void AddLayer(DenseLayer *layer);
	void AddLayer(int _backunits_len, int _units_len, double _learning_rate, bool _is_input_layer, DenseLayer::Activation t);
	void BuildLayer();
	void Summary();
	double Train(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, int _train_round, double _accuracy, int validnum);
	Eigen::MatrixXd Predict(const Eigen::MatrixXd& xdata, int output_len);
	void Compare(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, const Eigen::VectorXd& y_mean, const Eigen::VectorXd& y_std,int num);
	double Cal_loss(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata);
	// Add scalor
private:
	vector<DenseLayer*> layers;
	vector<double> train_mse;
	vector<double> valid_mse;
	Eigen::MatrixXd loss_gradient;
	int train_round;
	double accuracy;
};

BPNN::BPNN()
{

}

BPNN::~BPNN()
{
	for (auto layer : layers)
		delete layer;
}

void BPNN::AddLayer(DenseLayer *layer)
{
	layers.push_back(layer);
}

void BPNN::AddLayer(int _backunits_len, int _units_len, double _learning_rate = 0.03, bool _is_input_layer = false, DenseLayer::Activation t = DenseLayer::Sigmoid)
{
	DenseLayer *layer = new DenseLayer(_backunits_len, _units_len, _learning_rate, _is_input_layer, t);
	layers.push_back(layer);
}

void BPNN::BuildLayer()
{
	for (int i = 0; i<layers.size(); i++)
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
		cout << "-------------" << i << "th layer-------------" << endl;
		cout << "weight shape = " << layers[i]->getbackunits() << "*" << layers[i]->getunits() << endl;
	}
}

double BPNN::Train(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, int _train_round, double _accuracy, int validnum = 3)
{
	train_round = _train_round;
	accuracy = _accuracy;

	int n = xdata.rows();
	double loss = 0;
	double all_loss = 0;
	Eigen::MatrixXd _xdata;
	Eigen::MatrixXd _ydata;

	if (n != ydata.rows())
	{
		cout << "Bad input data!" << endl;
		return 0;
	}

	Eigen::MatrixXd xdatatrainraw = xdata.block(0, 0, xdata.rows() - validnum, xdata.cols());
	Eigen::MatrixXd ydatatrainraw = ydata.block(0, 0, ydata.rows() - validnum, ydata.cols());
	Eigen::MatrixXd xdatavalidraw = xdata.block(xdata.rows() - validnum, 0, validnum, xdata.cols());
	Eigen::MatrixXd ydatavalidraw = ydata.block(ydata.rows() - validnum, 0, validnum, ydata.cols());

// Normalize 
	Eigen::VectorXd x_col_mean = xdatatrainraw.colwise().mean();
    Eigen::MatrixXd x_dev = xdatatrainraw.rowwise() - x_col_mean.transpose();	
	Eigen::VectorXd x_std_dev = (x_dev.array().square().colwise().sum()/(xdatatrainraw.rows()-1)).sqrt();
	Eigen::MatrixXd xdatatrain = x_dev.array().rowwise() / x_std_dev.transpose().array();

	Eigen::VectorXd y_col_mean = ydatatrainraw.colwise().mean();
    Eigen::MatrixXd y_dev = ydatatrainraw.rowwise() - y_col_mean.transpose();	
	Eigen::VectorXd y_std_dev = (y_dev.array().square().colwise().sum()/(ydatatrainraw.rows()-1)).sqrt();
	Eigen::MatrixXd ydatatrain = y_dev.array().rowwise() / y_std_dev.transpose().array();

	Eigen::MatrixXd xdatavalid_dev = (xdatavalidraw.rowwise() - x_col_mean.transpose());
	Eigen::MatrixXd xdatavalid = xdatavalid_dev.array().rowwise() / x_std_dev.transpose().array();

	Eigen::VectorXd ydatavalid_col_mean = ydatavalidraw.colwise().mean();
    Eigen::MatrixXd ydatavalid_dev = ydatavalidraw.rowwise() - ydatavalid_col_mean.transpose();	
	Eigen::VectorXd ydatavalid_std_dev = (ydatavalid_dev.array().square().colwise().sum()/(ydatavalidraw.rows()-1)).sqrt();
	Eigen::MatrixXd ydatavalid = ydatavalid_dev.array().rowwise() / ydatavalid_std_dev.transpose().array();

// Normalize end

	cout << "Initial mse on training set is " << Cal_loss(xdatatrain, ydatatrain) << endl;
	Compare(xdatatrain, ydatatrain, y_col_mean, y_std_dev,3);

	cout << "Initial mse on validation set is " << Cal_loss(xdatavalid, ydatavalid) << endl;
	Compare(xdatavalid, ydatavalid, y_col_mean, y_std_dev, 3);

	for (int i = 0; i < train_round; i++)
	{
		all_loss = 0;
		for (int j = 0; j < xdatatrain.rows(); j++)
		{
			_xdata = xdatatrain.row(j);
			_ydata = ydatatrain.row(j);

			for (auto layer : layers)
			{
				_xdata = layer->ForwardPropagation(_xdata);
			}

			loss_gradient = 2.0 * (_xdata - _ydata);
			loss = loss_gradient.unaryExpr(std::ref(x_square_by_4)).sum();

			all_loss += loss;

			for (int k = 0; k < layers.size() - 1; k++)
			{
				loss_gradient = layers[layers.size() - 1 - k]->BackwardPropagation(loss_gradient);
			}
		}

		double mse = all_loss / xdatatrain.rows();
		train_mse.push_back(mse);
		/*if (abs(train_mse[train_mse.size() - 2] - train_mse[train_mse.size() - 1]) < accuracy)
		{
		cout << "Satisfy accuracy!" << endl;
		return mse;
		}*/
		double mse_valid = Cal_loss(xdatavalid, ydatavalid);
		valid_mse.push_back(mse_valid);

		cout<< "------------- Finished training round " << i << " -------------" << endl;
		cout <<"mse on training set is "<< mse << endl;
		cout << "mse on validation set is " << mse_valid << endl;

		// Normalize
		cout << "Training set example:" << endl;
		Compare(xdatatrain, ydatatrainraw, y_col_mean, y_std_dev, 3);

		cout << "Validation set example:" << endl;
		Compare(xdatavalid, ydatavalidraw, y_col_mean, y_std_dev, 3);
		// Normalize

		if (mse < accuracy)
		{
			cout << "Satisfy accuracy!" << endl;
			return mse;
		}

	}

	return 0;
}

double BPNN::Cal_loss(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata)
{
	double all_loss = 0;
	Eigen::MatrixXd _xdata(1, xdata.cols());
	for (int j = 0; j < xdata.rows(); j++)
	{
		all_loss += (Predict(xdata.row(j), ydata.cols()) - ydata.row(j)).unaryExpr(std::ref(x_square)).sum();
	}
	return all_loss / xdata.rows();
}

// Normalize
void BPNN::Compare(const Eigen::MatrixXd& xdata, const Eigen::MatrixXd& ydata, const Eigen::VectorXd& y_mean, const Eigen::VectorXd& y_std, int num)
{
	for (int k = 0; k < num; k++)
	{	
		Eigen::MatrixXd pred = (Predict(xdata.row(k), ydata.cols())).array().rowwise() * y_std.transpose().array();
		pred = pred.rowwise() + y_mean.transpose();

		cout << "Predict value of sample " << k << " is " << pred << ". ";
		cout << "Real value is " << ydata.row(k) << ". ";
		cout << "Difference is " << pred - ydata.row(k) << ". "<<endl;
	}
}

// Normalize end

Eigen::MatrixXd BPNN::Predict(const Eigen::MatrixXd& xdata, int output_len)
{
	Eigen::MatrixXd ydata(xdata.rows(), output_len);
	Eigen::MatrixXd _xdata;
	for (int j = 0; j < xdata.rows(); j++)
	{
		_xdata = xdata.row(j);
		for (auto layer : layers)
		{
			_xdata = layer->ForwardPropagation(_xdata);
		}
		for (int i = 0; i < output_len; i++)
			ydata(j, i) = _xdata(0, i);
	}
	return ydata;
}

void readdata(string filename, Eigen::MatrixXd& xdata, Eigen::MatrixXd& ydata)
{
	ifstream infile;
	infile.open(filename); 
	//assert(infile.is_open());

	string date;
	double data;
	for (int i = 0; i < stock_rows; i++)
	{
		infile >> date;
		//cout << date << endl;
		for (int j = 0; j < stock_cols; j++)
		{
			infile >> data;
			xdata(i,j)= data;
		}
		infile >> data;
		//cout << data << endl;
		ydata(i, 0) = data;
	}

	infile.close();

	cout << "Successfully read the data from " +filename+"!" << endl;
}

void real_train()
{
	Eigen::MatrixXd xdata(stock_rows, stock_cols);
	Eigen::MatrixXd ydata(stock_rows, 1);
	readdata("SPXdata.txt", xdata, ydata);

	double learning_rate = 0.0003;
	DenseLayer::Activation act_fun = DenseLayer::ReLu;
	BPNN modelnew;

	modelnew.AddLayer(stock_cols, stock_cols, learning_rate, true, act_fun);
	modelnew.AddLayer(stock_cols, 50, learning_rate, false, act_fun);
	modelnew.AddLayer(50, 5, learning_rate, false, act_fun);
	modelnew.AddLayer(5, 1, learning_rate, false, DenseLayer::None);
	modelnew.BuildLayer();
	modelnew.Summary();

	modelnew.Train(xdata, ydata, 250, 0.25, 30);
}

int main()
{
	//train on real data
	real_train();

	//test the training process
	//test();

	system("pause");
	return 0;
}

//main problem
//1. Gradient Exploding
// pretraining, cutting
//2. don't work, structrue is bad
// no solution
// yanerdaoling

// Scaling
// Yiyao Chen
// add scalor
// change Predict

// momentum
// Ludi Wang

// IOinfrastructure
// save load
// override: weights, valid result (validation result)
// append: learning curve
// model("name")
// "modelname_activation_layeri.txt"
// load


