""" Validate functions used to score association rules. """

import unittest

from gunter.apriori import (load_dataset, create_candidates, support_prune,
                            apriori, generate_rules)
from gunter.evaluate_association import lift, interest_factor

class TestEvaluateAssociation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Load data
        
        dataset = cls.load_dataset()
        D = map(set, dataset)

        # Derive association rules using Apriori

        F, support_data = apriori(dataset, min_support=0.01)
        H = generate_rules(F, support_data, min_confidence=0.01, verbose=False)

        # Assign class attributes

        cls.support_data = support_data
        cls.H = H
        cls.F = F
        cls.D = D

    @classmethod
    def load_dataset(cls):
        """Loads an example of market basket transactions for testing purposes.

        Returns
        -------
        A list (database) of lists (transactions). Each element of a transaction 
        is an item.
        """
        return [['Bread', 'Milk'], 
                ['Bread', 'Diapers', 'Beer', 'Eggs'], 
                ['Milk', 'Diapers', 'Beer', 'Coke'], 
                ['Bread', 'Milk', 'Diapers', 'Beer'], 
                ['Bread', 'Milk', 'Diapers', 'Coke']]


    def test_lift_score_example(self):

        X = self.support_data[frozenset(['Bread','Beer'])]
        Y = self.support_data[frozenset(['Diapers'])]

        # Support
        
        assert X == 0.4
        assert Y == 0.8

        # Confidence
        
        confidence = [i for i in self.H if i[0] == frozenset(['Bread', 'Beer']) \
                      and i[1] == frozenset(['Diapers'])][0][2]

        assert confidence == 1.0

        lift_score = confidence / (X * Y)
        
        assert  round(lift_score, 4) == 3.125
        
        # Positive correlation present

        assert lift_score > 1

    def test_lift_score_function(self):
        """ 
        Note that the function should generate identical results to the above example. 
        """

        output = lift(self.H, self.support_data)

        assert len(output) == 154

        check = [i for i in output if i[0] == frozenset(['Bread', 'Beer']) \
                 and i[1] == frozenset(['Diapers'])]

        # Check confidence
        
        assert check[0][2] == 1.0

        # Check lift score
        
        assert round(check[0][3], 4) == 3.125

        # Positive correlation present

        assert check[0][3] > 1
