/**
 * BayesianNode is an implementation of a single node in a bayesian network allowing
 * sampling from its conditional distribution.
 */
class BayesianNode {
    /**
     * @param {object} nodeDefinition - node structure and distributions definition
     *                                  taken from the network definition file
     */
    constructor(nodeDefinition) {
        this.nodeDefinition = nodeDefinition;
    }

    /**
     * Extracts unconditional probabilities of node values given the values of the parent nodes
     * @param {object} parentValues - values of the parent nodes
     * @private
     */
    _getProbabilitiesGivenKnownValues(parentValues = {}) {
        let probabilities = this.nodeDefinition.conditionalProbabilities;

        for (const parentName of this.parentNames) {
            const parentValue = parentValues[parentName];
            if (parentValue in probabilities.deeper) {
                probabilities = probabilities.deeper[parentValue];
            } else {
                probabilities = probabilities.skip;
            }
        }
        return probabilities;
    }

    /**
     * Randomly samples from the given values using the given probabilities
     * @param {array} possibleValues - a list of values to sample from
     * @param {number} totalProbabilityOfPossibleValues - a sum of probabilities of possibleValues
     *                                                    in the conditional distribution
     * @param {object} probabilities - a dictionary of probabilities from the conditional distribution
     *                                 indexed by the values
     * @private
     */
    _sampleRandomValueFromPossibilities(possibleValues, totalProbabilityOfPossibleValues, probabilities) {
        let chosenValue = possibleValues[0];
        const anchor = Math.random() * totalProbabilityOfPossibleValues;
        let cumulativeProbability = 0;
        for (const possibleValue of possibleValues) {
            cumulativeProbability += probabilities[possibleValue];
            if (cumulativeProbability > anchor) {
                chosenValue = possibleValue;
                break;
            }
        }

        return chosenValue;
    }

    /**
     * Randomly samples from the conditional distribution of this node given values of parents
     * @param {object} parentValues - values of the parent nodes
     */
    sample(parentValues = {}) {
        const probabilities = this._getProbabilitiesGivenKnownValues(parentValues);
        const possibleValues = Object.keys(probabilities);

        return this._sampleRandomValueFromPossibilities(possibleValues, 1.0, probabilities);
    }

    /**
     * Randomly samples from the conditional distribution of this node given restrictions on the possible
     * values and the values of the parents.
     * @param {object} parentValues - values of the parent nodes
     * @param {object} valuePossibilities - a list of possible values for this node
     * @param {object} bannedValues - what values of this node are banned
     */
    sampleAccordingToRestrictions(parentValues, valuePossibilities, bannedValues) {
        const probabilities = this._getProbabilitiesGivenKnownValues(parentValues);
        let totalProbability = 0.0;
        const validValues = [];
        const valuesInDistribution = Object.keys(probabilities);
        const possibleValues = valuePossibilities || valuesInDistribution;
        for (const value of possibleValues) {
            if (!bannedValues.includes(value) && valuesInDistribution.includes(value)) {
                validValues.push(value);
                totalProbability += probabilities[value];
            }
        }

        if (validValues.length === 0) return false;
        return this._sampleRandomValueFromPossibilities(validValues, totalProbability, probabilities);
    }

    get name() {
        return this.nodeDefinition.name;
    }

    get parentNames() {
        return this.nodeDefinition.parentNames;
    }
}

module.exports = {
    BayesianNode,
};
