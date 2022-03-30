# Generative bayesian network
NodeJs package containing a bayesian network capable of randomly sampling from a distribution defined by a json file.

<!-- toc -->

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [API Reference](#api-reference)

<!-- tocstop -->

## Installation

## Usage

## Examples

## API Reference
All public classes, methods and their parameters can be inspected in this API reference.

<a name="BayesianNetwork"></a>

### BayesianNetwork
BayesianNetwork is an implementation of a bayesian network capable of randomly sampling from the distribution
represented by the network.


* [BayesianNetwork](#BayesianNetwork)
    * [`new BayesianNetwork(networkDefinition)`](#new_BayesianNetwork_new)
    * [`.generateSample(inputValues)`](#BayesianNetwork+generateSample)
    * [`.generateConsistentSampleWhenPossible(valuePossibilities)`](#BayesianNetwork+generateConsistentSampleWhenPossible)


* * *

<a name="new_BayesianNetwork_new"></a>

#### `new BayesianNetwork(networkDefinition)`

| Param | Type | Description |
| --- | --- | --- |
| networkDefinition | <code>object</code> | object defining the network structure and distributions |


* * *

<a name="BayesianNetwork+generateSample"></a>

#### `bayesianNetwork.generateSample(inputValues)`
Randomly samples from the distribution represented by the bayesian network.
Can generate samples not found in the data used to create the definition file.


| Param | Type | Description |
| --- | --- | --- |
| inputValues | <code>object</code> | node values that are known already |


* * *

<a name="BayesianNetwork+generateConsistentSampleWhenPossible"></a>

#### `bayesianNetwork.generateConsistentSampleWhenPossible(valuePossibilities)`
Randomly samples from the distribution represented by the bayesian network.
Cannot generate samples not found in the data used to create the definition file,
so it only generates samples when it is possible to be consistent with the data.


| Param | Type | Description |
| --- | --- | --- |
| valuePossibilities | <code>object</code> | a dictionary of lists of possible values for nodes                                      (if a node isn't present in the dictionary, all values are possible) |


* * *

