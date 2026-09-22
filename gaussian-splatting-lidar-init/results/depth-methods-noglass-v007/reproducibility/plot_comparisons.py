"""Scientific depth and signed-error panels with fixed, shared colour scales."""
from pathlib import Path
import json,sys,os
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'matplotlib-cache')
sys.path.insert(0,str(ROOT/'plot-deps'))
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT.parents[2]/'3dgsResult/zs601-depth-methods-v007'
DEST=OUT/'comparison_figures';DEST.mkdir(exist_ok=False)
views=json.loads((OUT/'selected_views.json').read_text())
metrics=json.loads((OUT/'metrics/geometry_metrics.json').read_text())
rows={(r['image_id'],r['method']):r for r in metrics['per_view']}
depth_cmap=plt.get_cmap('viridis').copy();depth_cmap.set_bad('#10151f')
error_cmap=plt.get_cmap('RdBu_r').copy();error_cmap.set_bad('#10151f')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold'})

for v in views:
    name=v['name'];stem=Path(name).stem
    ref=np.load(OUT/'reference_noglass_mesh/depth_float'/f'{stem}.npy')
    a=np.asarray(Image.open(OUT/'method_a_gaussian/depth'/name)).astype(float)/1000
    b=np.asarray(Image.open(OUT/'method_b_zbuffer/depth'/name)).astype(float)/1000
    rgb=np.asarray(Image.open(OUT/'method_a_gaussian/images'/name))
    am=(a>0)&(ref>0);bm=(b>0)&(ref>0)
    ra=rows[(v['image_id'],'method_a_gaussian')];rb=rows[(v['image_id'],'method_b_zbuffer')]
    fig,ax=plt.subplots(2,3,figsize=(12,15))
    fig.subplots_adjust(top=.915,bottom=.14,left=.04,right=.98,wspace=.08,hspace=.12)
    for pane,z,title in zip(ax[0],[ref,a,b],['No-glass mesh first hit\n(reference only)',
        f'A: Gaussian depth\n{ra["coverage_percent"]:.2f}% valid',
        f'B: point z-buffer\n{rb["coverage_percent"]:.2f}% valid']):
        im=pane.imshow(np.ma.masked_where(z<=0,z),vmin=0,vmax=10,cmap=depth_cmap,interpolation='nearest')
        pane.set_title(title,fontsize=10)
    for pane,z,mask,title in zip(ax[1,:2],[a,b],[am,bm],['A: signed depth error','B: signed depth error']):
        err=pane.imshow(np.ma.masked_where(~mask,(z-ref)*1000),vmin=-100,vmax=100,cmap=error_cmap,interpolation='nearest')
        pane.set_title(title)
    ax[1,2].imshow(rgb,interpolation='nearest');ax[1,2].set_title('A: RGB (black background)')
    for pane in ax.flat:pane.set_xticks([]);pane.set_yticks([])
    cb=fig.colorbar(im,cax=fig.add_axes([.08,.075,.36,.013]),orientation='horizontal');cb.set_label('Camera Z (m), fixed 0-10 m')
    cb=fig.colorbar(err,cax=fig.add_axes([.58,.075,.36,.013]),orientation='horizontal',extend='both');cb.set_label('PNG Z - reference Z (mm)\nblue: in front | red: behind')
    fig.suptitle(f'View {stem} | same no-glass 1 cm cloud, same camera\n'
        f'Common-mask depth MAE: A {ra["common_valid_depth"]["mean_mm"]:.1f} mm | B {rb["common_valid_depth"]["mean_mm"]:.1f} mm',fontsize=13)
    fig.text(.5,.012,'Dark pixels: invalid. Depth/error panels use nearest-neighbor display. Error colours saturate at ±100 mm.\n'
        'A: k=3, scale 0.5, opacity 1 (float32), harmonic camera Z. B: one nearest point per occupied pixel; no hole filling.',
        ha='center',va='bottom',fontsize=8)
    fig.savefig(DEST/f'{stem}_depth_comparison.png',dpi=150,pad_inches=.15)
    plt.close(fig)
    print('PLOTTED',stem,flush=True)

A=metrics['aggregate']['method_a_gaussian'];B=metrics['aggregate']['method_b_zbuffer']
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,values,title,unit in zip(axes,
    [[A['coverage_percent'],B['coverage_percent']],
     [A['point_to_input']['mean_mm'],B['point_to_input']['mean_mm']],
     [A['common_valid_depth']['mean_mm'],B['common_valid_depth']['mean_mm']]],
    ['Depth valid coverage','Point-to-input NN (all own valid)','First-hit depth MAE (common mask)'],['%','mm','mm']):
    bars=ax.bar(['A: Gaussian','B: z-buffer'],values,color=['#2563eb','#e87b29'],width=.55)
    ax.set_title(title,fontsize=11);ax.set_ylabel(unit);ax.spines[['top','right']].set_visible(False)
    ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_ylim(0,max(values)*1.23)
fig.suptitle('10-view smoke | close to an input point does not prove correct visibility',fontsize=13)
fig.savefig(DEST/'summary_metrics.png',dpi=180)
plt.close(fig)
print('PLOTS_COMPLETE')
