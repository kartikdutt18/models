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
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

std::queue<std::string> batchNormRunningMean;
std::queue<std::string> batchNormRunningVar;
FFN<CrossEntropyError<>> model;

size_t inputWidth = 224;
size_t inputHeight = 224;

template <
    typename OutputLayer = mlpack::ann::CrossEntropyError<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void LoadWeights(mlpack::ann::FFN<OutputLayer, InitializationRule>& model,
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
    typename InitializationRule = mlpack::ann::RandomInitialization
>void HardCodedRunningMeanAndVariance(
    mlpack::ann::FFN<OutputLayer, InitializationRule> &model)
{
  arma::mat runningMean, runningVar;
  // vector<size_t> indices ={ 1, 2 };
  vector<size_t> indices ={1, 2};
  for (size_t idx : indices)
  {
      LoadBNMats(runningMean, runningVar);
      std::cout << "Loading RunningMean and Variance for " << idx << std::endl;
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[idx])->Model()[1])->TrainingMean() = runningMean.t();
      boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[idx])->Model()[1])->TrainingVariance() = runningVar.t();
  }

  // vector<size_t> darknet53Cfg ={ 1, 2, 8, 8, 4 };
  vector<size_t> darknet53Cfg ={1, 2, 8, 8, 4};

  size_t cnt = 3;
  for (size_t blockCnt : darknet53Cfg)
  {
      for (size_t layer = 0; layer < blockCnt; layer++)
      {
          std::cout << "Loading RunningMean and Variance for " << cnt << std::endl;
          LoadBNMats(runningMean, runningVar);

          std::cout << boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Residual<>*>(model.Model()[cnt])->Model()[0])->Model()[1])->InputSize() << " ---- " << runningMean.n_elem << std::endl;;
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Residual<>*>(model.Model()[cnt])->Model()[0])->Model()[1])->TrainingMean() = runningMean.t();
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Residual<>*>(model.Model()[cnt])->Model()[0])->Model()[1])->TrainingVariance() = runningVar.t();
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Residual<>*>(model.Model()[cnt])->Model()[0])->Model()[1])->Deterministic() = true;
          std::cout << "Loading RunningMean and Variance for " << cnt << std::endl;
          LoadBNMats(runningMean, runningVar);
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Residual<>*>(model.Model()[cnt])->Model()[1])->Model()[1])->TrainingMean() = runningMean.t();
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Residual<>*>(model.Model()[cnt])->Model()[1])->Model()[1])->TrainingVariance() = runningVar.t();
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(boost::get<Residual<>*>(model.Model()[cnt])->Model()[1])->Model()[1])->Deterministic() = true;
          cnt++;
      }

      if (blockCnt != 4)
      {
          std::cout << "Loading RunningMean and Variance for " << cnt << std::endl;
          LoadBNMats(runningMean, runningVar);
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[cnt])->Model()[1])->TrainingMean() = runningMean.t();
          boost::get<BatchNorm<>*>(boost::get<Sequential<>*>(model.Model()[cnt])->Model()[1])->TrainingVariance() = runningVar.t();
          cnt++;
      }
  }

  cout << batchNormRunningMean.size() << endl;
}

size_t ConvOutSize(const size_t size,
    const size_t k,
    const size_t s,
    const size_t padding)
{
    return std::floor(size + 2 * padding - k) / s + 1;
}

template<typename SequentialType = Sequential<>>
void ConvolutionBlock(const size_t inSize,
    const size_t outSize,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth = 1,
    const size_t strideHeight = 1,
    const size_t padW = 0,
    const size_t padH = 0,
    const bool batchNorm = true,
    SequentialType* baseLayer = NULL)
{
    Sequential<>* bottleNeck = new Sequential<>();
    bottleNeck->Add(new Convolution<>(inSize, outSize, kernelWidth,
        kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
        inputHeight));

    // Update inputWidth and input Height.
    std::cout << "Conv Layer.  ";
    std::cout << "(" << inputWidth << ", " << inputHeight <<
        ", " << inSize << ") ----> ";

    inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
    std::cout << "(" << inputWidth << ", " << inputHeight <<
        ", " << outSize << ")" << std::endl;

    if (batchNorm)
    {
        bottleNeck->Add(new BatchNorm<>(outSize, 1e-5, false));
    }

    bottleNeck->Add(new LeakyReLU<>(0.01));

    if (baseLayer != NULL)
    {
        baseLayer->Add(bottleNeck);
    }
    else
    {
        model.Add(bottleNeck);
    }
}

void DarkNet53ResidualBlock(const size_t inputChannel,
    const size_t kernelWidth = 3,
    const size_t kernelHeight = 3,
    const size_t padWidth = 1,
    const size_t padHeight = 1)
{
    std::cout << "Residual Block Begin." << std::endl;
    Residual<>* residualBlock = new Residual<>();
    ConvolutionBlock(inputChannel, inputChannel / 2,
        1, 1, 1, 1, 0, 0, true, residualBlock);
    ConvolutionBlock(inputChannel / 2, inputChannel, kernelWidth,
        kernelHeight, 1, 1, padWidth, padWidth, true, residualBlock);
    model.Add(residualBlock);
    std::cout << "Residual Block end." << std::endl;
}


int main()
{
  arma::mat input(224 * 224 * 3, 1), output;
  input.ones();

  model.Add<IdentityLayer<>>();
  ConvolutionBlock(3, 32, 3, 3, 1, 1, 1, 1, true);
  ConvolutionBlock(32, 64, 3, 3, 2, 2, 1, 1, true);

  size_t curChannels = 64;

  // Residual block configuration for DarkNet 53.
  std::vector<size_t> residualBlockConfig ={1, 2, 8, 8, 4};
  for (size_t blockCount : residualBlockConfig)
  {
      for (size_t i = 0; i < blockCount; i++)
      {
          DarkNet53ResidualBlock(curChannels);
      }

      if (blockCount != 4)
      {
          ConvolutionBlock(curChannels, curChannels * 2, 3, 3,
              2, 2, 1, 1, true);
          curChannels = curChannels * 2;
      }
  }

  model.Add<AdaptiveMeanPooling<>>(1, 1);
  model.Add<Linear<>>(1024, 1000);

  model.ResetParameters();
  model.Parameters().zeros();

  LoadWeights(model, "./../../../cfg/darknet53_features.xml");
  HardCodedRunningMeanAndVariance(model);

  model.Predict(input, output);
  arma::mat desiredOutput;
  mlpack::data::Load("./../../.././models/darknet53/mlpack-weights/linear_weight_158.csv", desiredOutput);
  output = model.Parameters().submat(arma::span(model.Parameters().n_elem - 1000 - 1024 * 1000,
      model.Parameters().n_elem - 1000 - 1), arma::span());
  /*
   * mlpack::data::Load("avg_pool_output.csv", desiredOutput);
  */
  for (int i = 0; i < output.n_elem; i++)
  {
    double diff = abs(output(i) - desiredOutput(i));
    if (diff != 0)
      cout << diff << " " << i << endl;
    if (i % 10 == 0)
      cout << "Checkpoint : " << i << endl;
  }
  return 0;
}