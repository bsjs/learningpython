# -*- coding: utf-8 -*-
import numpy as np
import pylab
class LinerRegression:
    'LinerRegression demo'
    #train the param theate1 theate2
    def compute_param(self,data,theate1,theate2,step,iter_num):
        #the num of iter
        for i in range(iter_num):
            [theate1,theate2] = self.compute_gradient(data,theate1,theate2,step)
            if i % 100 == 0:
                print('iter {0}:error={1}'.format(i, self.compute_error(data,theate1,theate2)))
        return [theate1,theate2];
    #compute the grad
    def compute_gradient(self,data,theate1,theate2,step):
        #compute the data length
        data_len = len(data);

        x = data[:,0];
        y = data[:,1];

        #compute the partial derivative
        theate1_grad = -(2/data_len)*(y-theate1-theate2*x);
        theate1_grad = np.sum(theate1_grad,axis=0);
        new_theate1 = theate1-step*theate1_grad;

        theate2_grad = -(2/data_len)*(y-theate1-theate2*x)*x;
        theate2_grad=np.sum(theate2_grad,axis=0);
        new_theate2 = theate2-step*theate2_grad;

        return [new_theate1,new_theate2]
    #compute the error
    def compute_error(self,data,theate1,theate2):
        total_error = 0;
        date_len = len(data);
        x = data[:,0];
        y = data[:,1];
        total_error = (theate1+theate2*x-y)**2/2;
        total_error = np.sum(total_error,axis=0);
        return total_error/float(date_len);
    #plot the data
    def plot_data(self,data,theate1,theate2):
        x = data[:,0];
        y = data[:,1];
        yp = theate1+theate2*x;
        pylab.plot(x,y,'o');
        pylab.plot(x,yp,'or-');
        pylab.show();
    #liner regression
    def liner_regression(self):
        #get the data
        data = np.loadtxt('data.csv',delimiter=',');
        #params: theate1 theate2 stepLength:learning length iter_num:iter
        theate1 = 0.0;
        theate2 = 0.0;
        step_length = 0.001;
        iter_num = 1000;
        [theate1,theate2] = self.compute_param(data,theate1,theate2,step_length,iter_num);
        #plot data
        self.plot_data(data,theate1,theate2);
if __name__ == '__main__':
    linner = LinerRegression();
    linner.liner_regression();


