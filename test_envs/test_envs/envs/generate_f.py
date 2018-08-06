import numpy as np
import sympy
qt_list = [0.84 ,0.87 ,0.9 , 0.92,0.94, 0.96, 0.98, 0.99, 0.995]

D_list =   [[0.0125,0.078, 0.0001, 0.0018, 0.9995], \
            [-0.0014, 0.0186, 0.0092, 0.019, 1.0049], \
            [-0.0096, -0.05, -0.0789, 0.0105, 0.9999], \
            [-0.011, -0.0254,-0.0227, 0.009, 1.0043], \
            [-0.0136,-0.0441, -0.0438,-0.0022,1.0012]]

def generate_f_dict(qt_list,D_list):
    F_dict = np.zeros([len(qt_list),len(D_list)])
    for qt_index,qt in enumerate(qt_list):
        for D_index,D in enumerate(D_list):
            F_dict[qt_index][D_index] = F(qt,D)
            print ("({},{})------qt:{} Dt:{} size:{}".format(qt_index,D_index,qt,D,F_dict[qt_index][D_index]))

def F (qt,Dt):
        x=sympy.Symbol('x')
        zero =- qt
        for i,d in enumerate(Dt[::-1]):
            zero+=d*x**i
        temp = (sympy.solve(zero,x))
        result = None
        for y in temp:
            if sympy.im(y) != 0:
                continue
            if -3< y and y<0:
                result = y
        if result is None:
            raise IOError("f cannot be commpute by qt--{} Dt--{}".format(qt,Dt))
        return np.array(result)
generate_f_dict(qt_list,D_list)