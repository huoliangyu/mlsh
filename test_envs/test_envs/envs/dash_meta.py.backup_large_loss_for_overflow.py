"""
1. all the elements in state need normalize to (0,1), so I divide the max value when return state
(in obs funtion ),respectively,
 and multiply the max value when input qt(in step funciton). This method construct on the assumption that the agent is not 
 sensitive for the scope of each input
 except self.phi
2.what if B>Bmax  now I add a very large punishment for B>Bmax
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import queue
import random
import math
import sympy
import time

logger = logging.getLogger(__name__)

is_normal = True
is_multi_Dt = False

qt_list = [0.84 ,0.87 ,0.9 , 0.92,0.94, 0.96, 0.98, 0.99, 0.995]
c_list = [0.5, 1, 2, 3, 4, 5, 6, 8, 10]
goal_list=[[5,2,50],[1,10,50],[1,2,250]] #default [1,2,50]
D_list =   [[0.0125,0.078, 0.0001, 0.0018, 0.9995], \
            [-0.0014, 0.0186, 0.0092, 0.019, 1.0049], \
            [-0.0096, -0.05, -0.0789, 0.0105, 0.9999], \
            [-0.011, -0.0254,-0.0227, 0.009, 1.0043], \
            [-0.0136,-0.0441, -0.0438,-0.0022,1.0012]]
class DashMeta(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # new action space = [0.84 ,0.87 ,0.9 , 0.92,0.94, 0.96, 0.98, 0.99, 0.995]
        self.action_space = spaces.Discrete(9)
        # self.realgoal = np.random.randint(0,2)
        # self.action = qt_list
        self._seed()
        self.alpha,self.beta,self.gamma = goal_list[0]
        self.Bmax  = 20
        self.Bth = 10
        self.episode_num = 400    #represent how many segments in oen episode
        self.viewer = None
        self.eps = 0.001
        self.T = 2
        self.fmax = 20
        self.p = 0.5
        self.generate_f_dict(qt_list,D_list)
        obs = self.reset()
        self.observation_space = spaces.Box(-10000000, 10000000, shape=(obs.shape[0],))
        # self.steps_beyond_done = None
        
        # Just need to initialize the relevant attributes
        self._configure()

    def randomizeCorrect(self):
        self.realgoal = self.np_random.randint(0,3)
        self.alpha,self.beta,self.gamma = goal_list[self.realgoal]
        # print("new goal is " + str(self.realgoal))

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        print("seeded")
        return [seed]

    def _step(self, action):
        action = qt_list[action]
        qt_1 = self.qt 
        self.qt = action
        f = self.F(self.qt, self.Dt)
        tao = f*1.0/self.c_real
        self.phi = max(0,tao-self.Bt)
        if tao>self.Bt:
            self.rebuffing_time+=self.phi
            self.rebuffing_num+=1
        self.Bt = self.T +max(0,self.Bt-tao)
        self.c_estimate.append(self.c_real)
        self.c_estimate = self.c_estimate[1:]
        self.Dt = self.D_queue.get()
        self.c_real = self.Markov(self.c_real)
        r_ssim = self.qt
        r_jitter = abs(self.qt - qt_1)
        r_rebuffing = self.phi
        r_future = max(0,self.Bth - self.Bt)^2
        r_overflow = 1 if self.Bt>self.Bmax else 0
        reward = self.alpha*r_ssim - self.beta*r_jitter - self.gamma* r_rebuffing -self.eps *r_future-1000000*r_overflow
        self.state = self.get_state()
        obs = self.obs()
        print (" now is step {}".format(self.cur_step))
        self.cur_step+=1
        if self.cur_step>= self.episode_num:
            self.terminal = 1
        final_state = [obs,reward,self.terminal,[self.rebuffing_time,self.rebuffing_num,r_ssim,r_jitter,r_rebuffing,r_overflow]]
        if self.terminal==1:
            self.reset()
        return final_state

    def obs(self):
        
        return np.reshape(np.array(self.state + self.goals), (-1,)) 

    def _reset(self):
        # self.randomizeCorrect()
        self.qt = qt_list[0]
        self.c_estimate = [c_list[0],c_list[0]]
        self.c_real = c_list[0]
        self.phi = 0
        self.Bt = self.T     #duration time for each segment
        self.state = self.get_state()
        self.goals = []
        for x in goal_list:
            self.goals+=x

        self.D_queue = self.create_D(self.episode_num+1)
        self.Dt = self.D_queue.get()
        self.cur_step = 0
        self.terminal = 0
        self.rebuffing_time = 0
        self.rebuffing_num = 0
        
        return self.obs()

    # def _render(self, mode='human', close=False):
    #     if close:
    #         if self.viewer is not None:
    #             self.viewer.close()
    #             self.viewer = None
    #         return

    #     screen_width = 400
    #     screen_height = 400


    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
    #         self.man_trans = rendering.Transform()
    #         self.man = rendering.make_circle(10)
    #         self.man.add_attr(self.man_trans)
    #         self.man.set_color(.5,.8,.5)
    #         self.viewer.add_geom(self.man)

    #         self.goal_trans = []
    #         for g in range(len(self.goals)):
    #             self.goal_trans.append(rendering.Transform())
    #             self.goal = rendering.make_circle(20)
    #             self.goal.add_attr(self.goal_trans[g])
    #             self.viewer.add_geom(self.goal)
    #             self.goal.set_color(.5,.5,g*0.8)


    #     self.man_trans.set_translation(self.state[0], self.state[1])
    #     for g in range(len(self.goals)):
    #         self.goal_trans[g].set_translation(self.goals[g][0], self.goals[g][1])

    #     return self.viewer.render(return_rgb_array = mode=='rgb_array')
    def get_state(self):
        if is_normal:
            state_c_list = [x/10.0 for x in self.c_estimate]
            return [self.qt] + state_c_list +[self.phi,self.Bt*1.0/self.Bmax]
        else:
            state_c_list = [x for x in self.c_estimate]
            return [self.qt] + state_c_list +[self.phi,self.Bt]
        
    def create_D(self,queue_length):
        q = queue.Queue()
        total_time =0
        while True:
            # get one D
            if is_multi_Dt:
                D_index = np.randint(0,5)
            else:
                D_index = 0
            D_list_selected = D_list[D_index]
            
            # get macrotime for selected D 
            _,n = math.modf(random.expovariate(1.0/10))
            
            D_time = int(n) if int(n)%2==0 else int(n)+1

            if total_time+ D_time >= queue_length:
                for _ in range(queue_length-total_time):
                    q.put(D_list_selected)
                break
            else:
                for _ in range(D_time):
                    q.put(D_list_selected)
                total_time+= D_time

        return q
    def F (self,qt,Dt):
        qt_index = qt_list.index(qt)
        Dt_index = D_list.index(Dt)
        result = self.F_dict[qt_index][Dt_index]
        return np.array(np.exp(result)*self.fmax)
    def Markov(self,c_cur):
        index = c_list.index(c_cur)
        eps = random.random()
        if eps<2.0*self.p/3:
            #both side
            if index>=1 and index< len(c_list)-1:
                if random.random()>0.5:
                    index_new = index+1
                else:
                    index_new = index-1
            elif index<1:
                index_new = index+1
            elif index== len(c_list)-1:
                index_new=index-1
            else:
                raise IOError("error c_index--{}".format(index))
        elif 2.0*self.p/3<eps and eps<self.p:
            if index>=2 and index< len(c_list)-2:
                if random.random()>0.5:
                    index_new = index+2
                else:
                    index_new = index-2
            elif index<2:
                index_new = index+2
            elif index>=len(c_list)-2:
                index_new=index-2
            else:
                raise IOError("error c_index--{}".format(index))
        else:
            index_new = index 
        return c_list[index_new]
    
    def generate_f_dict(self,qt_list,D_list):
        self.F_dict = np.zeros([len(qt_list),len(D_list)])
        for qt_index,qt in enumerate(qt_list):
            for Dt_index,Dt in enumerate(D_list):
                self.F_dict[qt_index][Dt_index] = self.GetF(qt,Dt)
                # print ("({},{})------qt:{} Dt:{} size:{}".format(qt_index,D_index,qt,D,F_dict[qt_index][D_index]))
        print ("generate F_dict done")
    def GetF(self,qt,Dt):
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