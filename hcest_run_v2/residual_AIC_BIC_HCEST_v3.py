import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import os
import sys
from PyPDF3 import PdfFileWriter, PdfFileReader, PdfFileMerger
from PyPDF3.pdf import PageObject
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Arial'
#rcParams['legend.title_fontsize'] = 15
sns.set_context('poster', font_scale=1.2)

# merge multi-page PDF into a single PDF
def merge_pdf(path, outputname):
    '''
    path is the path to the PDF file, including the pdf name
    outputname is the pdf file name of the output, not including .pdf
    '''
    reader = PdfFileReader(open(path,'rb'))

    # reader = PyPDF2.PdfFileReader(open("input.pdf",'rb'))

    NUM_OF_PAGES = reader.getNumPages()

    page0 = reader.getPage(0)
    h = page0.mediaBox.getHeight()
    w = page0.mediaBox.getWidth()

    newpdf_page = PageObject.createBlankPage(None, w, h*NUM_OF_PAGES)
    for i in range(NUM_OF_PAGES):
        next_page = reader.getPage(i)
        newpdf_page.mergeScaledTranslatedPage(next_page, 1, 0, h*(NUM_OF_PAGES-i-1))

    writer = PdfFileWriter()
    writer.addPage(newpdf_page)

    with open('{}.pdf'.format(outputname), 'wb') as f:
        writer.write(f)

def extract_HCEST_params(path):
    '''
    Combine individual CEST profile (pdf) into a big pdf file
    Combine individual fitparm.csv to a big csv file
    path is the path to the folders containing all fit results (i.e. fit_1, fit_2 etc)
    '''
    file_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    file_names = [name for name in file_names if name[:4] == 'fit_']
    # file_names = [name for name in file_names if len(name) == 5]
    file_names.sort(key = lambda x : int(x.split('_')[-1]))
    files_to_loop = filter(lambda x: x.startswith('fit_'), file_names)
    df_total = []
    pdf_total = []
    for i, n in enumerate(files_to_loop):
        df = pd.read_table(path + '/' + n + '/' + 'test/fitparms-result_{}.csv'.format(i+1), delimiter=',')
        df.columns = df.columns + '{}'.format(i+1)
        df_total.append(df)
        pdf_total.append(path + '/' + n + '/' + 'test/fit-result_{}.pdf'.format(i+1))

    df_final = pd.concat(df_total, axis=1)  
    df_final = df_final.drop(['param' + str(i+1) for i in range(len(df_total))[1:]], axis = 1)
    # df_final.columns = np.append(np.array(['']),np.repeat(np.array(assignment),2))
    df_final.loc[-1] = np.append(np.array(['']),np.repeat(np.array(assignment),2))
    df_final.index = df_final.index + 1
    df_final = df_final.apply(np.roll, shift=1)
    # df = df.sort_index()
#     df_final = df_final.set_index(list(df_final)[0])

    df_final.to_csv(path + '/' + name + '_fitparm.csv', index=False)
    
    # merge profiles
    merger = PdfFileMerger()

    for pdf in pdf_total:
        merger.append(pdf)

    merger.write(path + '/' + path.split('/')[-2] + '_result.pdf')
    merger.close()

def combine_pdf_hori(pdf_filenames, outputname):
    '''
    merge two pdf files horizontally
    pdf_filenames is the list of directory to two pdf files
    '''
    input1 = PdfFileReader(open(pdf_filenames[0], "rb"), strict=False)
    input2 = PdfFileReader(open(pdf_filenames[1], "rb"), strict=False)

    page1 = input1.getPage(0)
    page2 = input2.getPage(0)

    total_width = page1.mediaBox.upperRight[0] + page2.mediaBox.upperRight[0]
    total_height = max([page1.mediaBox.upperRight[1], page2.mediaBox.upperRight[1]])

    new_page = PageObject.createBlankPage(None, total_width, total_height)

    # Add first page at the 0,0 position
    new_page.mergePage(page1)
    # Add second page with moving along the axis x
    new_page.mergeTranslatedPage(page2, page1.mediaBox.upperRight[0], 0)

    output = PdfFileWriter()
    output.addPage(new_page)
    output.write(open(outputname, "wb"))


# Ref: http://www.originlab.com/doc/Origin-Help/PostFit-CompareFitFunc
# Ref:  Psychonomic Bulletin & Review 2004, 11 (1), 192-196

# Ref: http://www.originlab.com/doc/Origin-Help/PostFit-CompareFitFunc
# Ref:  Psychonomic Bulletin & Review 2004, 11 (1), 192-196

def AIC(N, exp, fit, K):
    '''
    calculate AIC given N(numbe of data), K(variables), exp:experimental intensity, fit:fitting intensity
    '''
    # residual sum of squares 
    RSS = np.sum((exp-fit)**2)
    
    if N/K >= 40:
        AIC = N*np.log(RSS/N) + 2*K
    else:
        AIC = N*np.log(RSS/N) + 2*K + 2*K*(K+1)/(N-K-1)
        
    return AIC

def BIC(N, exp, fit, K):
    '''
    calculate AIC given N(numbe of data), K(variables), exp:experimental intensity, fit:fitting intensity
    '''
    RSS = np.sum((exp-fit)**2)
    
    BIC = N*np.log(RSS/N) + K*np.log(N)
    
    return BIC

# larmor frequency
if str(sys.argv[4]) == '700':      
    lf = 699.9348710
elif str(sys.argv[4]) == '600':
    lf = 599.659943

def residual_HCEST2(path, path_noex, assignment, ol_num, name=None, ylim = [-0.12, 0.12], xticks = [-6, 6, 7]):
    '''
    draw residual distrubution plot for HCEST fitting with and without exchange
    path is the directory to the kex fitting folder
    the no kex fitting folder must be the named as path +"_no_kex"
    assignment is the list of peak assignment
    ol_num is the list of indexes for peaks that are overlapped
    '''
    num_row = len(assignment)//2+1
    fig = plt.figure(figsize = (24,2*num_row*10))
    count = 1
    #path_noex = path + '_no_kex'
    for i in range(1,len(assignment)+1):
        if i in ol_num:
            continue
        df_kex = pd.read_csv(path + '/fit_{0}/test/fit-result_{0}.csv'.format(i))
        
#         df_kex = df_kex[df_kex['norm_intensity']>0.1]
        
        df_nokex = pd.read_csv(path_noex + '/fit_{0}/test/fit-result_{0}.csv'.format(i))
        
#         df_nokex = df_nokex[df_nokex['norm_intensity']>0.1]
             
        number_slps = df_kex['slp(hz)'].unique()
        number_slps.sort()
        colors_plot = ['k', 'r', 'b', 'g', 'cyan', 'magenta', 'brown', 'yellow', 'teal', 'lightgreen']
        # calculate AIC/BIC
        kex_exp = df_kex['norm_intensity']
        kex_fit = df_kex['fit_norm_intensity']

        nokex_exp = df_nokex['norm_intensity']
        nokex_fit = df_nokex['fit_norm_intensity'] 

        AIC_kex = AIC(len(kex_exp), kex_exp.values, kex_fit.values, 5)
        BIC_kex = BIC(len(kex_exp), kex_exp.values, kex_fit.values, 5)

        AIC_nokex = AIC(len(nokex_exp), nokex_exp.values, nokex_fit.values, 2)
        BIC_nokex = BIC(len(nokex_exp), nokex_exp.values, nokex_fit.values, 2)

        AIC_total = np.array([AIC_kex, AIC_nokex])
        delta_AIC = AIC_total - np.min(AIC_total)
        wAIC_kex, wAIC_nokex = (np.exp(-0.5*delta_AIC)/np.sum(np.exp(-0.5*(delta_AIC))))

        BIC_total = np.array([BIC_kex, BIC_nokex])
        delta_BIC = BIC_total - np.min(BIC_total)
        wBIC_kex, wBIC_nokex = (np.exp(-0.5*delta_BIC)/np.sum(np.exp(-0.5*(delta_BIC))))      

        ax = fig.add_subplot(2*num_row,2,count)
        # Plot profile
        counter = 0
        for dummy_slp in number_slps:
            df_kex2 = df_nokex.loc[df_kex['slp(hz)'] == dummy_slp]
            plt.scatter(df_kex2['offset(hz)']/lf, df_kex2['norm_intensity'] - df_kex2['fit_norm_intensity'], color = colors_plot[counter], label='%4.0f'%float(dummy_slp))
            counter = counter + 1
            plt.xlim(np.min(df_kex2['offset(hz)']/lf)-0.8, np.max(df_kex2['offset(hz)']/lf)+0.8)
        ax.set_xticks(np.linspace(*xticks))
        ax.set_ylim(*ylim)
        ax.set_title('{}'.format(assignment[i-1]))
        
        l=ax.legend(loc = 4, fontsize = 30, frameon=False,
        handletextpad=-2.0, handlelength=0, markerscale=0,
        title = '$\omega$' + ' $2\pi$' + '$^{-1}$'+ ' (Hz)',
        ncol=3, columnspacing=1.8)
        l._legend_box.align = "right"
        for item in l.legendHandles:
            item.set_visible(False)
        for handle, text in zip(l.legendHandles, l.get_texts()):
            text.set_color(handle.get_facecolor()[0])
 
        
        ax = fig.add_subplot(2*num_row,2,count+1)
        # Plot profile
        counter = 0
        for dummy_slp in number_slps:
            df_kex1 = df_kex.loc[df_kex['slp(hz)'] == dummy_slp]
            plt.scatter(df_kex1['offset(hz)']/lf, df_kex1['norm_intensity'] - df_kex1['fit_norm_intensity'], color = colors_plot[counter], label='%4.0f'%float(dummy_slp))
            counter = counter + 1
            plt.xlim(np.min(df_kex1['offset(hz)']/lf)-0.8, np.max(df_kex1['offset(hz)']/lf)+0.8)
        ax.set_xticks(np.linspace(*xticks))
        ax.set_ylim(*ylim)
        ax.set_title('{}'.format(assignment[i-1]))
        line = '$wAIC$$_{+ex}$' + '= {:.2f}'.format(wAIC_kex) + '\n$wBIC$$_{+ex}$'+' ={:.2f}'.format(wBIC_kex)
        anchored_text = AnchoredText(line, loc=2, prop=dict(size=30), frameon= False)
        ax.add_artist(anchored_text)
        l=ax.legend(loc = 4, fontsize = 30, frameon=False,
        handletextpad=-2.0, handlelength=0, markerscale=0,
        title = '$\omega$' + ' $2\pi$' + '$^{-1}$'+ ' (Hz)',
        ncol=3, columnspacing=1.8)
        l._legend_box.align = "right"
        for item in l.legendHandles:
            item.set_visible(False)
        for handle, text in zip(l.legendHandles, l.get_texts()):
            text.set_color(handle.get_facecolor()[0])
        count = count+2
        
    plt.tight_layout()
    if name:
        plt.savefig(name, dpi = 300, transparent = True)
        

# sample nema
name = str(sys.argv[3])

# path to the folder containing fitting with exchange
path = os.getcwd() + '/' + str(sys.argv[1])

# path to the folder containing fitting without exchange
path_nokex = os.getcwd() + '/' + str(sys.argv[2])

# peak assignment
assignment = ['T5', 'T6', 'T7', 'T8/T4', 'T9', 'T22', 'G11', 'G10', 'G2']

# in the 3rd argument, specify the index of peaks that you want to skip in the AIC/BIC analysis
# adjust ylim
# adjust xticks if needed
residual_HCEST2(path, path_nokex, assignment, [], ylim=[-0.1, 0.1], 
    name = 'residual_{}_residual.pdf'.format(name), xticks = [-6, 6, 7])


# generate fitparm csv for fitting with exchange
extract_HCEST_params(path)
merge_pdf(path + '/' + path.split('/')[-2] + '_result.pdf',
         '{}_fit'.format(name))

# generate fitparm csv for fitting without exchange
extract_HCEST_params(path_nokex)
merge_pdf(path_nokex + '/' + path_nokex.split('/')[-2] + '_result.pdf',
         '{}_nokex'.format(name))

# combine all pdfs
pdf_filenames = ['{}_nokex.pdf'.format(name), '{}_fit.pdf'.format(name)]
combine_pdf_hori(pdf_filenames, '{}_combined.pdf'.format(name))

# remove redundant pdfs
os.system('rm ' + path + '/' + path.split('/')[-2] + '_result.pdf')
os.system('rm ' + path_nokex + '/' + path.split('/')[-2] + '_result.pdf')

