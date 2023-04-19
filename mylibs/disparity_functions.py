import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.colors import LogNorm
from skimage import img_as_ubyte
from skimage.color import rgb2gray

def SD_array(imageL, imageR, d_minimum, d_maximum):
    # initialization of the array of "squared differences" for different shifts
    SD = np.zeros((1+d_maximum-d_minimum,np.shape(imageL)[0],np.shape(imageL)[1]))
    for i in range(1+d_maximum-d_minimum):
        shifted_imageR = np.roll(imageR, i, axis=1)
        # do i need to change left end columns to 0 since rolled? 
        shifted_imageR[:,0:i+1] = 0
        SD[i] = np.square(np.linalg.norm(imageL.astype(float) - shifted_imageR.astype(float),axis=2))
    return SD

def integral_image(img):
    img_flt = img.astype(float)
    # cumulative sum across rows
    sum_rows = np.cumsum(img_flt, axis=1)
    # cumulative sum across columns
    sum_cols = np.cumsum(sum_rows, axis=0)
    return sum_cols

INFTY = np.inf

def windSum(img, window_width):
    int_img = integral_image(img)
    # define shifted int_img and fill new rows/cols with 0s 
    tr = np.roll(int_img, window_width, axis=0)
    tr[0:window_width, :] = 0
    bl = np.roll(int_img, window_width, axis=1)
    bl[:, 0:window_width] = 0
    tl = np.roll(tr, window_width, axis=1)
    tl[:, 0:window_width] = 0
    # sum the four images together for windowed sum 
    wind_sums = int_img - tr - bl + tl
    #reposition wind_sums as outlined in instruction
    margins = int((window_width-1)/2)
    wind_sums = np.roll(np.roll(wind_sums, -margins, axis=0), -margins, axis=1)
    # replace margins with infty
    firsts = [x for x in range(margins)]
    lasts = [-x-1 for x in range(margins)]
    wind_sums[firsts, :] = INFTY
    wind_sums[:, firsts] = INFTY
    wind_sums[lasts, :] = INFTY
    wind_sums[:, lasts] = INFTY
    return wind_sums

def SSDtoDmap(SSD_array, d_minimum, d_maximum):
    
    dMap = np.full(np.shape(SSD_array[0]),0)
    # initialize comparison with the first SSD array
    init = SSD_array[0]
    for i in range(d_maximum - d_minimum):
        # take the next SSD array to compare 
        new = SSD_array[i+1]
        # dMap updates if the new array has smaller element with corresponding disparity
        # i + 1 does NOT look like the correct value though, should be d_minimum + i + 1
        dMap = np.where(init <= new, dMap, d_minimum+i+1)
        # update init so that it is the element-wise minimum of all SSD arrays compared so far 
        init = np.where(init <= new, init, new)
    return dMap

def Dmap_Windows(imageL, imageR, d_minimum, d_maximum, window_width):
    SD_ars = SD_array(imageL, imageR, d_minimum, d_maximum)
    SSDw = np.zeros(np.shape(SD_ars))
    # find window sum for each disparity level
    for Delta in range(1+d_maximum-d_minimum):
        SSDw[Delta] = windSum(SD_ars[Delta],window_width)
    # find dmap from window sums 
    dmap = SSDtoDmap(SSDw, d_minimum, d_maximum)
    return dmap

def Viterbi(photo_const_mat, disp_array, v_mat):
    # input is SD_array for a scanline (SD_array[all_disparities, specific_row/scanline, all_columns]),
    # array of all possible disparity values, and matrix representing spatial coherence penalty
    n = photo_const_mat.shape[1]
    m = photo_const_mat.shape[0]
    # initialize E_bar with all 0s. We will calculate and store E_bar for each pixel sequentially
    E_bar = np.zeros((m,n))
    # something to keep track of the optimal edges
    # not exactly an adjacency matrix
    # at each column j+1, value is the row index 
    adj_mat = np.zeros((m,n))
    
    # perform forward pass
    for j in range(n-1):
        # previous pixel's E_bar. first pixel's E_bar is all 0s 
        prev_pxl_Ebar = E_bar[:,j]
        # photo_consistency cost as a part of E_j(d_j, d_j+1)
        prev_pxl_photo_const = photo_const_mat[:,j]
        # since photo_consistency and E_bar of prev pixel (j) is a vector that doesn't change, add them together
        x = prev_pxl_Ebar + prev_pxl_photo_const
        # matrix where each row i represents E_bar of j + photo_consistency of j + V(d_j, d_i)
        cur_pxl_Ebar = v_mat + x
        # the current pixel's E_bar is then row-wise minimum of the above matrix
        row_mins = cur_pxl_Ebar.min(1)
        E_bar[:,j+1] = row_mins
        # keep track of which disparity in prev_pixel gave the minimum for each current pixel disparity 
        # (in terms of index in disparity array)
        adj_mat[:,j+1] = np.argmin(cur_pxl_Ebar, axis=1)
    
    # backward pass
    optimal_disp_idx = np.array([0] * n)
    # check which disparity of the last pixel gave the minimum cumulative sum
    optimal_disp_idx[-1] = np.argmin(E_bar[:,-1])
    for j in range(n-1):
        # follow the edges recorded in adj_mat to get the 'shortest path'
        optimal_disp_idx[n-j-2] = adj_mat[optimal_disp_idx[n-j-1],n-j-1]
    optimal_disp = disp_array[optimal_disp_idx]
    # only need to return optimal_disp, rest for testing 
    return optimal_disp

def Viterbi_wrapper(imageL, imageR, d_minimum, d_maximum, w_param):
    # wrapper for Viterbi. prepares inputs for Viterbi and calls it for each scanline 
    SD = SD_array(imageL, imageR, d_minimum, d_maximum) 
    disp_array = np.array(range(d_minimum, d_maximum+1))
    v_mat = np.array([np.absolute(disp_array - disp_array[i]) for i in  range(len(disp_array))]) * w_param
    dmap = np.zeros(SD[0].shape)
    num_scanlines = dmap.shape[0]
    # find disparity map for each scanline using Viterbi
    for i in range(num_scanlines):
        dmap[i,:] = Viterbi(SD[:,i,:], disp_array, v_mat)
    return dmap

def Viterbi_wrapper_wind(imageL, imageR, d_minimum, d_maximum, w_param, window):
    # same as Viterbi wrapper, but using WindSum instead of SSD 
    SD = SD_array(imageL, imageR, d_minimum, d_maximum)
    num_del = (window-1) // 2
    # different shape due to infty being removed later 
    wind_SD = np.zeros((SD.shape[0], SD.shape[1] - (window - 1), SD.shape[2] - (window - 1)))
    for Delta in range(1+d_maximum-d_minimum):
        temp = windSum(SD[Delta],window)
        # need to get rid of inf entries or else the cost is inf due to roll no matter what 
        idx_del = [x for x in range(num_del)] + [-x-1 for x in range(num_del)]
        wind_SD[Delta] = np.delete(np.delete(temp, idx_del, 1), idx_del, 0)
    disp_array = np.array(range(d_minimum, d_maximum+1))
    v_mat = np.array([np.absolute(disp_array - disp_array[i]) for i in  range(len(disp_array))]) * w_param
    dmap = np.zeros(wind_SD[0].shape)
    num_scanlines = dmap.shape[0]
    # same as before, find disparity map for each scanline using Viterbi
    for i in range(num_scanlines):
        dmap[i,:] = Viterbi(wind_SD[:,i,:], disp_array, v_mat)
    return dmap

# update Viterbi wrapper so that it calculates V for each pixel pair in a scanline 
def Viterbi_wrapper_kernel(imageL, imageR, d_minimum, d_maximum, w_param, sigma):
    SD = SD_array(imageL, imageR, d_minimum, d_maximum) 
    disp_array = np.array(range(d_minimum, d_maximum+1))
    # v_mat will be changed for each scanline
    v_mat = np.array([np.absolute(disp_array - disp_array[i]) for i in  range(len(disp_array))])
    dmap = np.zeros(SD[0].shape)
    num_scanlines = dmap.shape[0]
    nd = len(disp_array)
    for i in range(num_scanlines):
        # change the v_mat input here for each scanline. follow the definition provided above 
        scanline = SD[:,i,:]
        # SSD in intensity for each disparity level 
        disp_int = np.array([scanline[j,j:j+nd] for j in range(nd)])
        # get wpq as a matrix
        wpq = w_param * np.exp(-disp_int/(2*sigma**2))
        # get vpq with element wise multiplication of v_mat and wpq
        vpq = np.multiply(v_mat, wpq)
        # use Viterbi to find disparity map for each scanline 
        dmap[i,:] = Viterbi(scanline, disp_array, vpq)
    return dmap

def Viterbi_wrapper_quad(imageL, imageR, d_minimum, d_maximum, w_param, sigma):
    SD = SD_array(imageL, imageR, d_minimum, d_maximum) 
    disp_array = np.array(range(d_minimum, d_maximum+1))
    # v_mat will be changed for each scanline
    v_mat = np.square(np.array([np.absolute(disp_array - disp_array[i]) for i in  range(len(disp_array))]))
    dmap = np.zeros(SD[0].shape)
    num_scanlines = dmap.shape[0]
    nd = len(disp_array)
    for i in range(num_scanlines):
        # change the v_mat input here for each scanline. follow the definition provided above 
        scanline = SD[:,i,:]
        # SSD in intensity for each disparity level 
        disp_int = np.array([scanline[j,j:j+nd] for j in range(nd)])
        # get wpq as a matrix
        wpq = w_param * np.exp(-disp_int/(2*sigma**2))
        # get vpq with element wise multiplication of v_mat and wpq
        vpq = np.multiply(v_mat, wpq)
        # use Viterbi to find disparity map for each scanline 
        dmap[i,:] = Viterbi(scanline, disp_array, vpq)
    return dmap

def Viterbi_wrapper_tquad(imageL, imageR, d_minimum, d_maximum, w_param, sigma, T):
    SD = SD_array(imageL, imageR, d_minimum, d_maximum) 
    disp_array = np.array(range(d_minimum, d_maximum+1))
    # v_mat will be changed for each scanline
    v_mat = np.square(np.array([np.absolute(disp_array - disp_array[i]) for i in  range(len(disp_array))]))
    T_mat = np.full(v_mat.shape, T)
    v_mat = np.minimum(v_mat, T_mat)
    dmap = np.zeros(SD[0].shape)
    num_scanlines = dmap.shape[0]
    nd = len(disp_array)
    for i in range(num_scanlines):
        # change the v_mat input here for each scanline. follow the definition provided above 
        scanline = SD[:,i,:]
        # SSD in intensity for each disparity level 
        disp_int = np.array([scanline[j,j:j+nd] for j in range(nd)])
        # get wpq as a matrix
        wpq = w_param * np.exp(-disp_int/(2*sigma**2))
        # get vpq with element wise multiplication of v_mat and wpq
        vpq = np.multiply(v_mat, wpq)
        # use Viterbi to find disparity map for each scanline 
        dmap[i,:] = Viterbi(scanline, disp_array, vpq)
    return dmap