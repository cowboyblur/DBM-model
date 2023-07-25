import random
import tkinter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import torch 
import torch.nn as nn
import time
from numpy.linalg import pinv
from matplotlib import cm


class DBM_Data:
    is_BD = 1  
    not_BD = 0
    rod = 2

    def __init__(self, length, width, root_pos,use_lightning_rod,lightning_rod_pos,lightning_rod_height):
        # Setting two more line in the row and column will make it easy to enable the periodic map
        self.column=length+2  
        self.row=width+2     
        self.map = np.zeros((self.row, self.column), dtype=int)
        self.map[1,root_pos] =DBM_Data.is_BD
        self.use_lightning_rod=use_lightning_rod
        self.lightning_rod_pos=lightning_rod_pos
        self.lightning_rod_height=lightning_rod_height
        self.U_cur=np.zeros((self.row, self.column),dtype=float)
        self.U_upd=np.zeros((self.row, self.column),dtype=float)
        for i in range(1,self.row-1):
            for j in range(1,self.column-1):
                self.U_cur[i,j]=100*((i-1)/(self.row-3))
        if self.use_lightning_rod:
            for i in range(lightning_rod_height):
                self.U_cur[self.row-2-i,lightning_rod_pos]=100
                self.map[self.row-2-i,lightning_rod_pos]=DBM_Data.rod

    def U_matrix_upd(self):
        #Calculate the best relaxation factor
        SOR_w=2/(1+np.sin(2*np.pi/(self.row+self.column-2)))
        delta=1
        times=0
        self.U_upd=self.U_cur.copy()
        while delta>1e-5:
            delta=0
            times=times+1
            for i in range(2,self.row-2):
                for j in range(2,self.column-2):
                    if self.map[i,j]==DBM_Data.not_BD:
                        self.U_upd[i,j]=self.U_cur[i,j]+SOR_w*\
                            (self.U_cur[i,j+1]+self.U_cur[i+1,j]+self.U_upd[i-1,j]\
                             +self.U_upd[i,j-1]-4*self.U_cur[i,j])/4
                        if abs(self.U_cur[i,j]-self.U_upd[i,j])>delta:
                            delta=abs(self.U_cur[i,j]-self.U_upd[i,j])
            self.U_cur=self.U_upd.copy()
    
    
    def Breakdown_Point(self):
        frac_para=3
        point_probability=np.zeros((self.row, self.column),dtype=float)
        #Calculate the whole map's potential probability
        whole_p=0
        for i in range(1,self.row-1):
            for j in range(1,self.column-1):
                if self.map[i,j]==DBM_Data.is_BD:
                    if self.map[i-1,j]==DBM_Data.is_BD:
                        p_1=0
                    else:
                        p_1=(self.U_cur[i-1,j])**(frac_para)
                    if self.map[i,j-1]==DBM_Data.is_BD:
                        p_2=0
                    else:
                        p_2=(self.U_cur[i,j-1])**(frac_para)
                    if self.map[i-1,j-1]==DBM_Data.is_BD:
                        p_3=0
                    else:
                        p_3=(self.U_cur[i-1,j-1]/np.sqrt(2))**(frac_para)
                    if self.map[i+1,j]==DBM_Data.is_BD:
                        p_4=0
                    else:
                        p_4=(self.U_cur[i+1,j])**(frac_para)
                    if self.map[i,j+1]==DBM_Data.is_BD:
                        p_5=0
                    else:
                        p_5=(self.U_cur[i,j+1])**(frac_para)
                    if self.map[i+1,j+1]==DBM_Data.is_BD:
                        p_6=0
                    else:
                        p_6=(self.U_cur[i+1,j+1]/np.sqrt(2))**(frac_para)
                    if self.map[i-1,j+1]==DBM_Data.is_BD:
                        p_7=0
                    else:
                        p_7=(self.U_cur[i-1,j+1]/np.sqrt(2))**(frac_para)
                    if self.map[i+1,j-1]==DBM_Data.is_BD:
                        p_8=0
                    else:
                        p_8=(self.U_cur[i+1,j-1]/np.sqrt(2))**(frac_para)
                    whole_p=whole_p+p_1+p_2+p_3+p_4+p_5+p_6+p_7+p_8
        #Calculate each point's breakdown probability
        s_1=0;s_2=0;s_3=0;s_4=0;s_5=0;s_6=0;s_7=0;s_8=0
        ran_choice=random.uniform(0,1)
        for i in range(1,self.row-1):
            for j in range(1,self.column-1):
                if self.map[i,j]==DBM_Data.is_BD:
                    if self.map[i-1,j]==DBM_Data.is_BD:
                        point_probability[i-1,j]=0
                    else:
                        point_probability[i-1,j]=(self.U_cur[i-1,j])**(frac_para)/whole_p
                    if self.map[i,j-1]==DBM_Data.is_BD:
                        point_probability[i,j-1]=0
                    else:
                        point_probability[i,j-1]=(self.U_cur[i,j-1])**(frac_para)/whole_p
                    if self.map[i-1,j-1]==DBM_Data.is_BD:
                        point_probability[i-1,j-1]=0
                    else:
                        point_probability[i-1,j-1]=(self.U_cur[i-1,j-1]/np.sqrt(2))**(frac_para)/whole_p
                    if self.map[i+1,j]==DBM_Data.is_BD:
                        point_probability[i+1,j]=0
                    else:
                        point_probability[i+1,j]=(self.U_cur[i+1,j])**(frac_para)/whole_p
                    if self.map[i,j+1]==DBM_Data.is_BD:
                        point_probability[i,j+1]=0
                    else:
                        point_probability[i,j+1]=(self.U_cur[i,j+1])**(frac_para)/whole_p
                    if self.map[i+1,j+1]==DBM_Data.is_BD:
                        point_probability[i+1,j+1]=0
                    else:
                        point_probability[i+1,j+1]=(self.U_cur[i+1,j+1]/np.sqrt(2))**(frac_para)/whole_p
                    if self.map[i-1,j+1]==DBM_Data.is_BD:
                        point_probability[i-1,j+1]=0
                    else:
                        point_probability[i-1,j+1]=(self.U_cur[i-1,j+1]/np.sqrt(2))**(frac_para)/whole_p
                    if self.map[i+1,j-1]==DBM_Data.is_BD:
                        point_probability[i+1,j-1]=0
                    else:
                        point_probability[i+1,j-1]=(self.U_cur[i+1,j-1]/np.sqrt(2))**(frac_para)/whole_p
                    s_1=s_8+point_probability[i-1,j]
                    s_2=s_1+point_probability[i,j-1]
                    s_3=s_2+point_probability[i-1,j-1]
                    s_4=s_3+point_probability[i+1,j]
                    s_5=s_4+point_probability[i,j+1]
                    s_6=s_5+point_probability[i+1,j+1]
                    s_7=s_6+point_probability[i-1,j+1]
                    s_8=s_7+point_probability[i+1,j-1]
                    #Decide which point would be broken down
                    intervals=[0,s_1,s_2,s_3,s_4,s_5,s_6,s_7,s_8,1]
                    interval_index=np.searchsorted(intervals,ran_choice,side='left')-1
                    if interval_index!=8:
                      if interval_index==0:
                        self.map[i-1,j]=DBM_Data.is_BD
                        self.U_cur[i-1,j]=0
                      if interval_index==1:
                        self.map[i,j-1]=DBM_Data.is_BD
                        self.U_cur[i,j-1]=0
                      if interval_index==2:
                        self.map[i-1,j-1]=DBM_Data.is_BD
                        self.U_cur[i-1,j-1]=0
                      if interval_index==3:
                        self.map[i+1,j]=DBM_Data.is_BD
                        self.U_cur[i+1,j]=0
                      if interval_index==4:
                        self.map[i,j+1]=DBM_Data.is_BD
                        self.U_cur[i,j+1]=0
                      if interval_index==5:
                        self.map[i+1,j+1]=DBM_Data.is_BD
                        self.U_cur[i+1,j+1]=0
                      if interval_index==6:
                        self.map[i-1,j+1]=DBM_Data.is_BD
                        self.U_cur[i-1,j+1]=0
                      if interval_index==7:
                        self.map[i+1,j-1]=DBM_Data.is_BD
                        self.U_cur[i+1,j-1]=0
                      break
            else:
                continue
            break
                    
    #If the flash reach the boundary,the system is over
    def is_to_boundary(self):
        is_to_boundary=False
        for i in range(1,self.column-1):
            if self.map[self.row-2,i]==1 :
                is_to_boundary=True
        for i in range(1,self.column-1):
            if self.map[i,1]==1 or self.map[i,self.column-2]==1:
                is_to_boundary=True
        if self.use_lightning_rod:
            for i in range(self.lightning_rod_height):
                if self.map[self.row-2-i,self.lightning_rod_pos]==DBM_Data.is_BD:
                    is_to_boundary=True
        return is_to_boundary

# Settings for the visual interface
class DBMTkVisual:
    kWindowZoomRatio = 0.9
    def __init__(self, array_data, DBM_length, DBM_width):
        self.data_row_size = len(array_data)     
        self.data_column_size = len(array_data[0])
        self.array_data = array_data
        self.color_dict = {DBM_Data.not_BD:'darkblue', DBM_Data.is_BD:'yellow',DBM_Data.rod:"red"}  
        self.root = tkinter.Tk()
        self.root.title('DBM for Flash')            
        size_window_length = \
          self.root.winfo_screenwidth() * DBMTkVisual.kWindowZoomRatio
        size_window_width = \
          self.root.winfo_screenheight() * DBMTkVisual.kWindowZoomRatio
        # Calculate the single pixel size
        pixel_length = size_window_length / float(DBM_length + 2)
        pixel_width = size_window_width / float(DBM_width + 2)
        pixel_size = min(pixel_length, pixel_width)
        self.pixel_size = int(pixel_size)
        # Update the window size
        size_window_length = size_window_length * self.pixel_size / pixel_length
        size_window_width = size_window_width * self.pixel_size / pixel_width
        windows_position_info = \
          '%dx%d+%d+%d' %(size_window_length, size_window_width, 
                         (self.root.winfo_screenwidth() - size_window_length)/2, 
                         (self.root.winfo_screenheight() - size_window_width)/2)
        self.root.geometry(windows_position_info) 
        self.canvas = tkinter.Canvas(self.root, bg = 'black')
        self.init_canvas()
        self.canvas.pack()   
        self.canvas.config(width=size_window_length, height=size_window_width)

    def init_canvas(self):
        # A 2d list for saving every handle of 'canvas rectangle'
        self.grid_handle = list() 
        self.canvas.create_rectangle(0, 0, self.pixel_size*(self.data_row_size), 
                                       self.pixel_size*(self.data_column_size),
                                       fill='gray',
                                       outline='gray')                
        # Plot every pixel in the canvas by 'for' cycle
        for row_num in range(1, self.data_row_size-1):                    
            self.grid_handle.insert(row_num-1,list())
            for column_num in range(1, self.data_column_size-1):            
                self.grid_handle[row_num-1].insert(column_num-1,
                self.canvas.create_rectangle(
                  self.pixel_size*column_num, 
                  self.pixel_size*row_num,
                  self.pixel_size*(column_num+1), 
                  self.pixel_size*(row_num+1),
                  fill=self.color_dict[self.array_data[row_num][column_num]],
                  outline=self.color_dict[self.array_data[row_num][column_num]]))

    def plot_pixel(self):
        # Use itemconfig to change the color of every canvas rectangle
        # Plot every pixel in the canvas by 'for' cycle and update
        for row_num in range(1, self.data_row_size-1):                
            for column_num in range(1, self.data_column_size-1):        
                self.canvas.itemconfig(
                  self.grid_handle[row_num-1][column_num-1],
                  fill=self.color_dict[self.array_data[row_num][column_num]],
                  outline=self.color_dict[self.array_data[row_num][column_num]]) 
        self.canvas.update()

def int_input_consider_default(show_str, defalut_value):
    # integer input reads considering the default value
    input_str = str(input(show_str))
    if '' == input_str:
      is_int_num = False
    else:
      is_int_num = True
    for str_element in range(len(input_str)):
        is_int_num = is_int_num and (ord(input_str[str_element])>=ord('0') \
                                      and ord(input_str[str_element])<=ord('9'))
        if not is_int_num:
           break
    if is_int_num:
        get_value = int(input_str)
    else:
        get_value = defalut_value
    return get_value

#user commands to create the model
def player_guide_interface():
    while True:
        print("Do you want to show each step of the evolution?")
        print("Y:show the every step on the GUI")
        print("N:just save the result into file ")
        input_str_1 = str(input('>>> '))
        data_file_mode = False
        if ('N' == input_str_1) or ('n' == input_str_1):
            data_file_mode = True
        #defalut value set
        dbm_data_length=100
        dbm_data_width=100
        root_pos=50
        use_lightning_rod=True
        lightning_rod_pos=28
        lightning_rod_height=30
        print('   ')
        print(' DEFAULT PARAMETER LIST:')
        print('| 2D Data Array Width:      %d' % dbm_data_width)
        print('| 2D Data Array Length:     %d' % dbm_data_length) 
        print('| Init Root Position:       %d' % root_pos)
        print('| Init Lightning rod position:       %d' % lightning_rod_pos)
        print('| Init Lightning rod height:       %d' % lightning_rod_height)
        print('Change the DEFAULT parameter?(Y/[N])')
        change_defult_parameter = str(input('>>> '))
        if ('Y' == change_defult_parameter) or  ('y' == change_defult_parameter):
            print("Set the Canvas:")
            print("Please input the 2D data array width:")
            dbm_data_width = int_input_consider_default('>>> ', dbm_data_width)
            print("Please input the 2D data array length:")
            dbm_data_length = int_input_consider_default('>>> ', dbm_data_length)
            print("Set the root of the flash:")
            root_pos=int_input_consider_default('>>>',root_pos)
            if root_pos>dbm_data_width-1:
                print("out of area!")
            print("Do you want to set a lightning rod?[Y/N]")
            input_str_2=str(input('>>>'))
            if ('N' == input_str_2) or ('n' == input_str_2):
                use_lightning_rod = False
            else:
                use_lightning_rod = True
                print("Please input the position of the rod")
                lightning_rod_pos=int_input_consider_default('>>>',lightning_rod_pos)
                print("Please input the height of the rod")
                lightning_rod_height=int_input_consider_default('>>>',lightning_rod_height)
        print("Start to calculate...")
        return data_file_mode,dbm_data_length,dbm_data_width,root_pos,use_lightning_rod,\
          lightning_rod_pos,lightning_rod_height

def U_show(U_array,len,wid):
    def title_and_labels(ax,title):
        ax.set_title(title);ax.set_xlabel("$y$")
        ax.set_ylabel("$x$");ax.set_zlabel("$U$")
    fig, axes=plt.subplots(2, 2, figsize=(6, 6),
                        subplot_kw={'projection': '3d'})
    y=np.linspace(2,wid-3,num=wid-4).astype('int32')
    x=np.linspace(2,len-3,num=len-4).astype('int32')
    X,Y=np.meshgrid(x,y)
    z=[]
    for i in range(0,len-4):
        for j in range(0,wid-4):
            z.append(U_array[x[i],y[j]])
    Z=np.array(z).reshape(len-4,wid-4)
    norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
    p=axes[0, 0].plot_surface(X,Y,Z,linewidth=0,rcount=20,
                            ccount=20,norm=norm,
                            cmap=cm.RdYlBu_r, edgecolor='blue')
    cb=fig.colorbar(p,ax=axes[0, 0],pad=0.1,shrink=0.6)
    title_and_labels(axes[0, 0], "surface plot")
    p=axes[0, 1].plot_wireframe(X,Y,Z,rcount=20,ccount=20,
                              color="green")
    title_and_labels(axes[0, 1],"wireframe plot")
    cset=axes[1, 0].contour(X,Y,Z,zdir='x',levels=20,
                          norm=norm,cmap=cm.RdYlBu_r, edgecolor='blue')
    title_and_labels(axes[1, 0],"contour x")

    cset=axes[1, 1].contour(X,Y,Z,zdir='y',levels = 20,
                          norm=norm,cmap=cm.RdYlBu_r, edgecolor='blue')
    title_and_labels(axes[1, 1],"contour y")
    fig.tight_layout(); plt.show()


def FileSave(U_array,len,wid,num):
    y=np.linspace(2,wid-3,num=wid-4).astype('int32')
    x=np.linspace(2,len-3,num=len-4).astype('int32')
    X,Y=np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("$y$")
    ax.set_ylabel("$x$")
    ax.set_zlabel("$U$")
    z=[]
    for i in range(0,len-4):
        for j in range(0,wid-4):
            z.append(U_array[x[i],y[j]])
    Z=np.array(z).reshape(len-4,wid-4)
    norm = mpl.colors.Normalize(vmin = Z.min(), vmax = Z.max())
    p=ax.plot_surface(X,Y,Z,linewidth=0,rcount=20,
                            ccount=20,norm=norm,
                            cmap=cm.RdYlBu_r, edgecolor='blue')
    cb=fig.colorbar(p,ax=ax,pad=0.1,shrink=0.6)
    plt.savefig('U_pic_%d.png' % num)
    plt.close(fig)


def main():
    file_mode,length, width, root_pos,use_lightning_rod,\
        lightning_rod_pos,lightning_rod_height=player_guide_interface()
    data=DBM_Data(length, width, root_pos,use_lightning_rod,\
        lightning_rod_pos,lightning_rod_height)
    plot=DBMTkVisual(data.map,length, width)
    flag=0
    runtime=time.time()
    while not data.is_to_boundary():
        sys.stdout.flush()
        sys.stdout.write("\b"*len("Iteration times:"+str(flag)))
        data.Breakdown_Point()
        data.U_matrix_upd()
        if not file_mode:
            plot.plot_pixel()
        flag=flag+1
        sys.stdout.write("Iteration times:"+str(flag))
        if file_mode:
            if flag%50==0:
                FileSave(data.U_cur,length,width,flag)
    FileSave(data.U_cur,length,width,flag)
    print("\n run time:%f" % (time.time()-runtime))
    U_show(data.U_cur,length,width)

if __name__ == '__main__':
  main()