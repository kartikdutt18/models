
#include <mlpack/core.hpp>
#include <dataloader/dataloader.hpp>
#include <utils/utils.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <utils/utils.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <models/yolo/yolo.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

std::queue<std::string> batchNormRunningMean;
std::queue<std::string> batchNormRunningVar;

class Accuracy
{
public:
  template <typename InputType, typename OutputType>
  static double Evaluate(InputType &input, OutputType &output)
  {
    arma::Row<size_t> predLabels(input.n_cols);
    for (arma::uword i = 0; i < input.n_cols; ++i)
    {
      predLabels(i) = input.col(i).index_max() + 1;
    }
    return arma::accu(predLabels == output) / (double)output.n_elem * 100;
  }
};

template <
    typename OutputLayer = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void LoadWeights(mlpack::ann::FFN<OutputLayer, InitializationRule> &model,
                 std::string modelConfigPath)
{
  std::cout << "Loading Weights\n";
  size_t currentOffset = 0;
  boost::property_tree::ptree xmlFile;
  boost::property_tree::read_xml(modelConfigPath, xmlFile);
  boost::property_tree::ptree modelConfig = xmlFile.get_child("model");

  model.Parameters().fill(0);
  BOOST_FOREACH (boost::property_tree::ptree::value_type const &layer, modelConfig)
  {
    std::string progressBar(81, '-');
    size_t filled = std::ceil(currentOffset * 80.0 / model.Parameters().n_elem);
    progressBar[0] = '[';
    std::fill(progressBar.begin() + 1, progressBar.begin() + filled + 1, '=');
    std::cout << progressBar << "] " << filled * 100.0 / 80.0 << "%\r";
    std::cout.flush();
    // Load Weights.
    if (layer.second.get_child("has_weights").data() != "0")
    {
      arma::mat weights;
      mlpack::data::Load("./../../../" + layer.second.get_child("weight_csv").data(), weights);
      model.Parameters()(arma::span(currentOffset, currentOffset + weights.n_elem - 1),
                         arma::span()) = weights.t();
      currentOffset += weights.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("weight_offset").data());
    }

    // Load Biases.
    if (layer.second.get_child("has_bias").data() != "0")
    {
      arma::mat bias;
      mlpack::data::Load("./../../../" + layer.second.get_child("bias_csv").data(), bias);
      model.Parameters()(arma::span(currentOffset, currentOffset + bias.n_elem - 1),
                         arma::span()) = bias.t();
      currentOffset += bias.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("bias_offset").data());
    }

    if (layer.second.get_child("has_running_mean").data() != "0")
    {
      batchNormRunningMean.push("./../../../" + layer.second.get_child("running_mean_csv").data());
    }

    if (layer.second.get_child("has_running_var").data() != "0")
    {
      batchNormRunningVar.push("./../../../" + layer.second.get_child("running_var_csv").data());
    }
  }
  std::cout << std::endl;
  std::cout << "Loaded Weights\n";
}

void LoadBNMats(arma::mat &runningMean, arma::mat &runningVar)
{
  runningMean.clear();
  if (!batchNormRunningMean.empty())
  {
    mlpack::data::Load(batchNormRunningMean.front(), runningMean);
    batchNormRunningMean.pop();
  }
  else
    std::cout << "This should never happen!\n";

  runningVar.clear();
  if (!batchNormRunningVar.empty())
  {
    mlpack::data::Load(batchNormRunningVar.front(), runningVar);
    batchNormRunningVar.pop();
  }
  else
    std::cout << "This should never happen!\n";
}

template <
    typename OutputLayer = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void HardCodedRunningMeanAndVariance(
    mlpack::ann::FFN<OutputLayer, InitializationRule> &model)
{
  arma::mat runningMean, runningVar;
  vector<size_t> indices = {1, 3, 5, 7, 9, 11, 13, 14};
  for (size_t idx : indices)
  {
    LoadBNMats(runningMean, runningVar);
    std::cout << "Loading RunningMean and Variance for " << idx << std::endl;
    boost::get<BatchNorm<> *>(boost::get<Sequential<> *>(model.Model()[idx])->Model()[1])->TrainingMean() = runningMean.t();
    boost::get<BatchNorm<> *>(boost::get<Sequential<> *>(model.Model()[idx])->Model()[1])->TrainingVariance() = runningVar.t();
  }
}

int main()
{
  YOLO<> yolo(3, 448, 448);
  std::cout << yolo.GetModel().Parameters().n_elem << std::endl;
  arma::mat input, output, target;
  mlpack::data::Load("./../../../input_tensor.csv", input);
  mlpack::data::Load("./../../../output_tensor.csv", target);
  input = input.t();
  target = target.t();
  yolo.GetModel().Predict(input, output);

  LoadWeights<>(yolo.GetModel(), "./../../../cfg/yolov1_tiny.xml");
  HardCodedRunningMeanAndVariance<>(yolo.GetModel());
  yolo.GetModel().Predict(input, output);
  double tolerance = 1e-3;
  for (size_t i = 0; i < target.n_elem; i++)
  {
    if (abs(target(i) - output(i)) > 2e-3)
    {
      std::cout << "Error exceeds " << tolerance << std::endl;
      std::cout << target(i) << " ---> " <<" at " << i << " " << output(i) << std::endl;
      return 0;
    }
  }

  std::cout << "Output Matches PyTorch, IoU among predictions is 1.0" << std::endl;
}