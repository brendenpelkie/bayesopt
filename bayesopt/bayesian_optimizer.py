
import numpy as np


class BayesianOptimizer:
    """
    Main bayesian optimizer instance
    """
    def __init__(self, oracle, acquisition_function, model, starter_data, bounds, grid_density = 101):
        # asssert starter data is a tuple of arrays, x and y, and have same length
        self.original_data = starter_data
        self.oracle_data = None
        self.all_data = None
        self.oracle = oracle
        self.acquisition_func = acquisition_function
        self.model = model
        # assert number of bounds is same shape as starter data
        self.bounds = bounds
        self.grid_density = grid_density


    def optimization_campaign(self, n_iterations, metrics):
        """
        run a Bayesian Optimization campaign
        """
    
        self.all_data = self.original_data

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
            querypts = self.acquisition_func(self.model, self.all_data, possible_points)
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
            #6. compute metrics 

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


        # remove points that we have already sampled
        # this code is slow. Should figure out something less naive. 

        existing_inds = []

        X, y = self.all_data

        for i in range(X.shape[0]):
            
            loc = np.where((X[i,:] == points).all(axis = 1))[0]
            if len(loc) == 1:
                    existing_inds.append(loc[0])
                    break
            
            
        points = np.delete(points, existing_inds, axis=0)

        return points.reshape(-1, len(self.bounds))






    

    