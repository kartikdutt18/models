/**
 * @file imagenette_trainer.hpp
 * @author Kartik Dutt
 *
 * Contains implementation of object classification suite. It can be used
 * to select object classification model, it's parameter dataset and
 * other training parameters.
 *
 * NOTE: This code needs to be adapted as this implementation doesn't support
 *       Command Line Arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <dataloader/dataloader.hpp>
#include <models/models.hpp>
#include <utils/utils.hpp>
#include <ensmallen_utils/print_metric.hpp>
#include <ensmallen_utils/periodic_save.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <utils/utils.hpp>
#include <ensmallen_utils/print_metric.hpp>
#include <ensmallen_utils/periodic_save.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>

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
  template<typename InputType, typename OutputType>
  static double Evaluate(InputType& input, OutputType& output)
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
}

void LoadBNMats(arma::mat& runningMean, arma::mat& runningVar)
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
    typename InitializationRule = mlpack::ann::RandomInitialization
>void HardCodedRunningMeanAndVariance(
    mlpack::ann::FFN<OutputLayer, InitializationRule> &model)
{
  arma::mat runningMean, runningVar;
  vector<size_t> indices = {1, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23};
  for (size_t idx : indices)
  {
    LoadBNMats(runningMean, runningVar);
    std::cout << "Loading RunningMean and Variance for " << idx << std::endl;
    boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[idx])->Model()[1])->TrainingMean() = runningMean.t();
    boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[idx])->Model()[1])->TrainingVariance() = runningVar.t();
  }
}

int main()
{
  DarkNet<mlpack::ann::NegativeLogLikelihood<>> darknet(3, 224, 224, 1000);
  LoadWeights<mlpack::ann::NegativeLogLikelihood<>>(darknet.GetModel(), "./../../../cfg/darknet19.xml");
  HardCodedRunningMeanAndVariance<mlpack::ann::NegativeLogLikelihood<>>(darknet.GetModel());
  DataLoader<> dataloader;
  dataloader.LoadImageDatasetFromDirectory("./../../../../imagenette-test/", 320, 320, 3,
      true, 0.3, false, {"resize : 224"});
  constexpr double RATIO = 0.4;
  constexpr size_t EPOCHS = 4;
  constexpr double STEP_SIZE = 0.001;
  constexpr int BATCH_SIZE = 8;

  mlpack::data::MinMaxScaler scaler;
  scaler.Fit(dataloader.TrainFeatures());
  scaler.Transform(dataloader.TrainFeatures(), dataloader.TrainFeatures());
  scaler.Transform(dataloader.ValidFeatures(), dataloader.ValidFeatures());
  std::cout << "Data scaled!\n";
  arma::mat predictions;
  darknet.GetModel().Predict(dataloader.TrainFeatures(), predictions);
  arma::Row<size_t> predLabels(predictions.n_cols);
  for (arma::uword i = 0; i < predictions.n_cols; ++i)
  {
      predLabels(i) = predictions.col(i).index_max() + 1;
  }
  predLabels.print();

  SGD<AdamUpdate> optimizer(STEP_SIZE, BATCH_SIZE,
  dataloader.TrainLabels().n_cols * EPOCHS,
      1e-8,
      true,
      AdamUpdate(1e-8, 0.9, 0.999));

  std::cout << "Optimizer Created, Starting Training!" << std::endl;
  dataloader.TrainLabels() = dataloader.TrainLabels() + 1;
  std::cout << "HERE" << std::endl;
  return 0;
  darknet.GetModel().Train(dataloader.TrainFeatures(),
      dataloader.TrainLabels(),
      optimizer,
      ens::PrintLoss(),
      ens::ProgressBar(),
      ens::EarlyStopAtMinLoss(),
      ens::PrintMetric<FFN<mlpack::ann::NegativeLogLikelihood<>, RandomInitialization>,
          Accuracy>(
            darknet.GetModel(),
            dataloader.TrainFeatures(),
            dataloader.TrainLabels(),
            "accuracy",
            true),
      ens::PrintMetric<FFN<mlpack::ann::NegativeLogLikelihood<>, RandomInitialization>,
          Accuracy>(
              darknet.GetModel(),
              dataloader.ValidFeatures(),
              dataloader.ValidLabels(),
              "accuracy",
              false),
      ens::PeriodicSave<FFN<mlpack::ann::NegativeLogLikelihood<>, RandomInitialization>>(
          darknet.GetModel(),
          "./../weights/",
          "darknet19", 1));

  mlpack::data::Save("darknet19.bin", "darknet",
      darknet.GetModel(), false);
  return 0;
}