
import numpy as np


class BayesianOptimizer:
    """
    Main bayesian optimizer instance
    """
    def __init__(self, oracle, acquisition_function, model, starter_data, method = 'max', valid_points = None, bounds = None, grid_density = 101):
        """
        Bayesian optimization object.

        Use the optimization_campaign method to run an optimization campaign

        Parameters:
        ---------
        oracle: Trained scitkit learn estimator that acts as the ground truth
        acquisition_function: function from bayesopt acquisitions module
        model: instantiated model from bayesopt model module. Currenlty on GPR supported
        starter_data (tuple, (X,y)): initial data to use for first iteration. 
        method: whether to max or min objective function 
        valid_points, bounds, grid_density: args for in-house valid point enumeration.

        
        """
        assert isinstance(starter_data, tuple), "Pass starterdata as a tuple (X,y)"
        assert len(starter_data[0]) == len(starter_data[1]), 'X and y have different dimensions'
        assert valid_points.shape[1] == starter_data[0].shape[1], 'Valid points have a different dimensionality than X'

        self.original_data = starter_data
        self.oracle_data = None
        self.all_data = None
        self.oracle = oracle
        self.acquisition_func = acquisition_function
        self.model = model
        self.bounds = bounds
        self.grid_density = grid_density
        self.available_points = valid_points
        self.method = method


    def optimization_campaign(self, n_iterations, metrics = None):
        """
        run a Bayesian Optimization campaign

        Paramters:
        ----------
        n_iterations: number of cycles to run

        Returns:
        --------
        results_dict: dictionary of relevant results for each iteration
        """
    
        self.all_data = self.original_data
        if self.available_points is None:
            if self.bounds is None:
                AssertionError('Must pass a set of available points in parameter space, or bounds on parameter space')
            else:
                self.available_points = self.enumerate_points()

        results_dict = {}

        for i in range(n_iterations):
            print(f'Starting iteration {i}')
            #1. Train model on current set of data
            print('Updating model')
            self.model.update(self.all_data)
            #2. get set of available points to sample from
            print('Getting possible points')
            possible_points = self.enumerate_points()
            #3. Evaluate acquisition function
            print('Calling acquisition function')
            querypts = self.acquisition_func(self)
            #4. query oracle
            print('Asking the oracle')
            oracle_results = self.oracle.predict(querypts)
            #5. update data with new result
            self.all_data = self.update_data(self.all_data, querypts, oracle_results)
            try:
                self.oracle_data = self.update_data(self.oracle_data, querypts, oracle_results)
            except TypeError:
                self.oracle_data = (querypts, oracle_results)

            results = {'querypts':querypts, 'oracleresult':oracle_results}
            results_dict[str(i)] = results
            #6. update the set of available points

            self.update_available_points()

        return results_dict


    def update_data(self, old_data, new_X, new_y):
        """
        handle apppending logic
        """

        X, y = old_data
        new_X = np.concatenate((X, new_X), axis = 0)
        new_y = np.concatenate((y, new_y), axis = 0)

        return (new_X, new_y)
    
    def enumerate_points(self):
        """
        Enumerate the points that are available for sampling
        """
        ranges = []
        for bounds in self.bounds:
            ranges.append(np.linspace(bounds[0], bounds[1], self.grid_density))
            
        points = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(self.bounds))

        return points.reshape(-1, len(self.bounds))
    
    def update_available_points(self):
        """
        Update the set of available points by removing ones that have been sampled
        """
        points = self.available_points
        existing_inds = []
        X, y = self.all_data
        for i in range(X.shape[0]):
            loc = np.where((X[i,:] == points).all(axis = 1))[0]
            if len(loc) == 1:
                    existing_inds.append(loc[0])
                    break
            
            
        points = np.delete(points, existing_inds, axis=0)
        self.available_points = points






    

    