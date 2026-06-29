from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import fitz
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def odd(v):
    v = int(round(v))
    return v if v % 2 else v + 1

def median(vals, default=0.0):
    if not vals:
        return default
    return float(np.median(np.asarray(vals, dtype=np.float64)))

def render_pdf(pdf_path: Path, page_index=0, dpi=300, max_long=3000):
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    nat_h, nat_w = bgr.shape[:2]
    if max(nat_w, nat_h) > max_long:
        s = max_long / max(nat_w, nat_h)
        bgr = cv2.resize(bgr, (round(nat_w*s), round(nat_h*s)), interpolation=cv2.INTER_AREA)
    return bgr, nat_w, nat_h

def grayscale_weighted(bgr):
    b,g,r = cv2.split(bgr)
    gray = ((r.astype(np.uint16)*77 + g.astype(np.uint16)*150 + b.astype(np.uint16)*29) >> 8).astype(np.uint8)
    return gray

def sauvola(gray, k=0.18, R=128):
    h,w = gray.shape
    win = clamp(odd(round(min(w,h)*0.01)), 15, 51)
    if win % 2 == 0: win += 1
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win,win), normalize=True, borderType=cv2.BORDER_REFLECT)
    sqmean = cv2.boxFilter(f*f, cv2.CV_32F, (win,win), normalize=True, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(0.0, sqmean - mean*mean))
    thr = mean * (1.0 + k*(std/R - 1.0))
    return (f < thr).astype(np.uint8), win

def histo_peak(hist, min_key=1):
    if len(hist) <= min_key:
        return {'main':min_key,'count':0,'lo':min_key,'hi':min_key}
    main = min_key + int(np.argmax(hist[min_key:]))
    main_count = int(hist[main])
    floor = max(1, int(round(main_count*0.25)))
    lo=hi=main
    gap=0
    k=main-1
    while k>=min_key:
        if hist[k] >= floor:
            lo=k; gap=0
        else:
            gap += 1
            if gap>1: break
        k-=1
    gap=0
    k=main+1
    while k<len(hist):
        if hist[k] >= floor:
            hi=k; gap=0
        else:
            gap += 1
            if gap>1: break
        k+=1
    return {'main':main,'count':main_count,'lo':lo,'hi':hi}

def estimate_scale(bin_img):
    h,w = bin_img.shape
    max_black = max(8, round(h*0.08))
    max_combo = max(16, round(h*0.25))
    black_hist = np.zeros(max_black+1, dtype=np.int64)
    white_hist = np.zeros(max_combo+1, dtype=np.int64)
    # every column, exact run lengths
    for x in range(w):
        col = bin_img[:,x]
        changes = np.flatnonzero(np.diff(np.r_[0,col,0]))
        # for binary 1 runs: pairs changes[0], changes[1], etc
        for j in range(0, len(changes), 2):
            if j+1 >= len(changes): break
            s,e = int(changes[j]), int(changes[j+1])
            L=e-s
            if 1 <= L <= max_black: black_hist[L]+=1
        # white gaps between black runs
        if len(changes)>=4:
            runs=[(int(changes[j]),int(changes[j+1])) for j in range(0,len(changes)-1,2)]
            for (_,e1),(s2,_) in zip(runs,runs[1:]):
                L=s2-e1
                if 2 <= L <= max_combo: white_hist[L]+=1
    bp=histo_peak(black_hist,1)
    wp=histo_peak(white_hist,2)
    staff_h = bp['main'] if bp['count']>0 else 2
    valid_white = wp['count'] >= max(8, int(w*0.01))
    staff_space = wp['main'] + staff_h if valid_white else 0
    scale_class='vertical_runs'
    # sanity: spacing should be notably larger than thickness and realistic
    if staff_space < max(4, staff_h*2) or staff_space > h*0.05:
        row = bin_img.sum(axis=1).astype(np.float32)
        positives = row[row>0]
        p75 = float(np.percentile(positives,75)) if positives.size else 0
        thr=max(w*0.2,p75*0.5)
        peaks=[]
        for y in range(1,h-1):
            if row[y]>=thr and row[y]>=row[y-1] and row[y]>row[y+1]:
                if not peaks or y-peaks[-1]>=3:
                    peaks.append(y)
                elif row[y]>row[peaks[-1]]:
                    peaks[-1]=y
        gaps=[b-a for a,b in zip(peaks,peaks[1:]) if 3 <= b-a <= h*0.05]
        staff_space=median(gaps, max(6,h/120))
        scale_class='row_fallback'
    return float(staff_h), float(staff_space), scale_class, bp, wp

def adaptive_line_response(gray, bin_img):
    h,w=gray.shape
    inv=(255-gray).astype(np.uint8)
    klen=clamp(round(w*0.048),55,140)
    hor=cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(klen,1)))
    bg=hor.astype(np.float32)
    for _ in range(3):
        bg=cv2.boxFilter(bg, cv2.CV_32F, (1,25), normalize=True, borderType=cv2.BORDER_REFLECT)
    sharp=np.maximum(0, hor.astype(np.float32)-bg)
    sample=sharp[::4,::4]
    pos=sample[sample>0]
    p20=float(np.percentile(pos,20)) if pos.size else 8.0
    thr=clamp(p20*0.9,6,16)
    mask=(sharp>=thr).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,1)))
    line_ink=np.maximum(bin_img,(mask>0).astype(np.uint8))
    return line_ink, mask, float(thr), int(klen)

@dataclass
class Track:
    pts: List[Tuple[float,float,float]] = field(default_factory=list) # x,y,strength
    slope: float=0.0
    missed:int=0
    support:float=0.0
    closed:bool=False
    def pred(self,x):
        if not self.pts: return 0
        lx,ly,_=self.pts[-1]
        return ly + self.slope*(x-lx)
    def add(self,x,y,v):
        if self.pts:
            lx,ly,_=self.pts[-1]
            dx=x-lx
            ns=(y-ly)/dx if dx else 0
            if len(self.pts)>=2:
                self.slope=0.6*self.slope+0.4*ns
            else:
                self.slope=ns
        self.pts.append((x,y,v)); self.support+=v; self.missed=0

def strip_peaks(line_ink, staff_h, staff_space):
    h,w=line_ink.shape
    strip_px=max(24,round(staff_space*10))
    N=clamp(round(w/strip_px),8,64)
    min_sep=max(2,staff_space*0.55)
    smooth_r=max(0,round(staff_h/2))
    out=[]
    for s in range(N):
        x0=round(s*w/N); x1=round((s+1)*w/N)
        if x1<=x0: continue
        cx=(x0+x1-1)/2
        proj=line_ink[:,x0:x1].sum(axis=1).astype(np.float32)
        if smooth_r>0:
            sm=cv2.boxFilter(proj.reshape(-1,1), cv2.CV_32F, (1,2*smooth_r+1), normalize=False, borderType=cv2.BORDER_REPLICATE).ravel()
        else: sm=proj
        thr=max((x1-x0)*0.3, float(sm.max())*0.4)
        cand=[]
        for y in range(1,h-1):
            if sm[y]>=thr and sm[y]>=sm[y-1] and sm[y]>sm[y+1]:
                lo=max(0,y-smooth_r-1); hi=min(h,y+smooth_r+2)
                weights=proj[lo:hi]
                sw=float(weights.sum())
                yy=float((weights*np.arange(lo,hi)).sum()/sw) if sw>0 else float(y)
                cand.append((yy,float(sm[y])))
        cand.sort(key=lambda p:p[1], reverse=True)
        chosen=[]
        for p in cand:
            if all(abs(p[0]-q[0])>=min_sep for q in chosen):
                chosen.append(p)
        chosen.sort(key=lambda p:p[0])
        out.append((cx,chosen))
    return N,out

def link_tracks(peaks_by_strip, staff_space, N):
    tol=max(2,staff_space*0.6)
    active=[]; finished=[]
    for cx,peaks in peaks_by_strip:
        used=[False]*len(peaks)
        # process stronger/longer tracks first
        for tr in sorted(active, key=lambda t:(len(t.pts),t.support), reverse=True):
            pred=tr.pred(cx)
            best=None; bestd=1e9
            for i,(y,v) in enumerate(peaks):
                if used[i]: continue
                d=abs(y-pred)
                if d<=tol and d<bestd:
                    best=i; bestd=d
            if best is not None:
                y,v=peaks[best]; used[best]=True; tr.add(cx,y,v)
            else:
                tr.missed+=1
        still=[]
        for tr in active:
            if tr.missed>2:
                finished.append(tr)
            else:
                still.append(tr)
        active=still
        for i,(y,v) in enumerate(peaks):
            if not used[i]:
                tr=Track(); tr.add(cx,y,v); active.append(tr)
    finished.extend(active)
    min_pts=max(3,round(N*0.4))
    lines=[]
    for idx,tr in enumerate(finished):
        if len(tr.pts)<min_pts: continue
        pts=np.array([[p[0],p[1]] for p in tr.pts],dtype=np.float64)
        lines.append({'id':idx,'pts':pts,'meanY':float(np.mean(pts[:,1])),'x0':float(pts[0,0]),'x1':float(pts[-1,0]),'support':float(tr.support),'strength':float(tr.support/len(tr.pts))})
    return lines

def ls_slope(pts):
    if len(pts)<2: return 0.0
    x=pts[:,0]; y=pts[:,1]
    xm=x.mean(); den=((x-xm)**2).sum()
    return float(((x-xm)*(y-y.mean())).sum()/den) if den>1e-9 else 0.0

def y_at(line,x):
    pts=line['pts']; xs=pts[:,0]; ys=pts[:,1]
    slope=line.get('slope',ls_slope(pts))
    if x<=xs[0]: return float(ys[0]+slope*(x-xs[0]))
    if x>=xs[-1]: return float(ys[-1]+slope*(x-xs[-1]))
    j=int(np.searchsorted(xs,x))
    x0,x1=xs[j-1],xs[j]; y0,y1=ys[j-1],ys[j]
    t=(x-x0)/(x1-x0) if x1!=x0 else 0
    return float(y0+t*(y1-y0))

def extend_line(line, ink, staff_h, staff_space):
    h,w=ink.shape
    half=max(1,round(staff_h/2)+1)
    band=half+max(2,round(staff_space*0.3))
    gap_tol=max(6,round(staff_space*1.2))
    line['slope']=ls_slope(line['pts'])
    support=np.zeros(w,dtype=np.uint8)
    for x in range(w):
        y=round(y_at(line,x)); lo=max(0,y-band); hi=min(h,y+band+1)
        support[x]=1 if ink[lo:hi,x].any() else 0
    best_s=best_e=0; cur_s=0; last_good=-1; best_len=0
    for x,val in enumerate(support):
        if val:
            last_good=x
            if x-last_good>gap_tol: cur_s=x
        if last_good>=0 and x-last_good>gap_tol:
            e=last_good
            if e-cur_s+1>best_len: best_s,best_e,best_len=cur_s,e,e-cur_s+1
            cur_s=x+1
            last_good=-1
    if last_good>=0:
        e=last_good
        if e-cur_s+1>best_len: best_s,best_e,best_len=cur_s,e,e-cur_s+1
    if best_len<=0: return line
    interior=[p for p in line['pts'] if best_s < p[0] < best_e]
    new=[(float(best_s),y_at(line,best_s))]+[(float(p[0]),float(p[1])) for p in interior]+[(float(best_e),y_at(line,best_e))]
    line['pts']=np.asarray(new,dtype=np.float64)
    line['x0']=float(best_s); line['x1']=float(best_e); line['meanY']=float(np.mean(line['pts'][:,1])); line['length']=float(best_e-best_s)
    return line

def dedupe_lines(lines, spacing):
    # retain strongest among near-identical y trajectories
    lines=sorted(lines,key=lambda l:l['support'],reverse=True)
    kept=[]
    for l in lines:
        if any(abs(l['meanY']-k['meanY'])<0.35*spacing and abs(l.get('slope',0)-k.get('slope',0))<0.03 for k in kept):
            continue
        kept.append(l)
    return sorted(kept,key=lambda l:l['meanY'])

def five_line_score(group, spacing, relaxed=False):
    ys=[l['meanY'] for l in group]
    gaps=np.diff(ys)
    if len(gaps)==0: return None
    med=float(np.median(gaps)); maxerr=float(np.max(np.abs(gaps-med)))
    if relaxed:
        lo=.35*spacing; hi=2.0*spacing; errmax=max(8,1.1*spacing)
    else:
        lo=.42*spacing; hi=1.75*spacing; errmax=max(5,.72*spacing)
    if not (lo<=med<=hi and maxerr<=errmax): return None
    support=sum(l['support'] for l in group)
    length=sum(l.get('length',l['x1']-l['x0']) for l in group)
    return support+length+(1e6 if len(group)==5 else 0)

def group_staves(lines, spacing):
    lines=sorted(lines,key=lambda l:l['meanY'])
    clusters=[]; cur=[]
    for l in lines:
        if cur and l['meanY']-cur[-1]['meanY']>1.7*spacing:
            clusters.append(cur); cur=[]
        cur.append(l)
    if cur: clusters.append(cur)
    staves=[]; orphans=[]
    for c in clusters:
        # find valid consecutive 5 windows greedily
        i=0
        while i+5<=len(c):
            g=c[i:i+5]
            if five_line_score(g,spacing,False) is not None:
                staves.append(g); i+=5
            else:
                orphans.append(c[i]); i+=1
        orphans.extend(c[i:])
    # orphan recovery: search best 5 then 4
    orphans=sorted(orphans,key=lambda l:l['meanY'])
    used=set()
    candidates=[]
    for n in (5,4):
        for i in range(len(orphans)-n+1):
            g=orphans[i:i+n]
            score=five_line_score(g,spacing,n==4)
            if score is not None:
                candidates.append((score,n,i,g))
    candidates.sort(reverse=True,key=lambda t:t[0])
    for score,n,i,g in candidates:
        ids=[id(x) for x in g]
        if any(k in used for k in ids): continue
        staves.append(g); used.update(ids)
    rem=[l for l in orphans if id(l) not in used]
    return staves, rem

def complete_staves(staves):
    for g in staves:
        if len(g)<4: continue
        x0s=np.array([l['x0'] for l in g]); x1s=np.array([l['x1'] for l in g])
        tx0=float(np.percentile(x0s,10)); tx1=float(np.percentile(x1s,90))
        for l in g:
            pts=l['pts']; mids=[p for p in pts if tx0 < p[0] < tx1]
            l['pts']=np.asarray([(tx0,y_at(l,tx0))]+[(float(p[0]),float(p[1])) for p in mids]+[(tx1,y_at(l,tx1))],dtype=np.float64)
            l['x0']=tx0;l['x1']=tx1;l['length']=tx1-tx0;l['meanY']=float(np.mean(l['pts'][:,1]))
    return staves

def detect_stafflines(bgr):
    gray=grayscale_weighted(bgr)
    bin_img,win=sauvola(gray)
    staff_h,staff_space,scale_class,bp,wp=estimate_scale(bin_img)
    line_ink,resp,resp_thr,klen=adaptive_line_response(gray,bin_img)
    N,peaks=strip_peaks(line_ink,staff_h,staff_space)
    lines=link_tracks(peaks,staff_space,N)
    lines=[extend_line(l,line_ink,staff_h,staff_space) for l in lines]
    lines=[l for l in lines if l.get('length',0)>0]
    # refine spacing using plausible adjacent gaps
    ys=sorted(l['meanY'] for l in lines)
    gaps=[b-a for a,b in zip(ys,ys[1:]) if 2 <= b-a <= staff_space*1.6]
    if gaps: staff_space=float(np.median(gaps))
    lines=dedupe_lines(lines,staff_space)
    staves,orphans=group_staves(lines,staff_space)
    staves=complete_staves(staves)
    # slope/short filtering at stave level
    all_group_lines=[l for g in staves for l in g]
    ref_len=median([l.get('length',l['x1']-l['x0']) for g in staves if len(g)>=4 for l in g], median([l.get('length',0) for l in lines],1))
    slopes=np.array([abs(l.get('slope',0)) for l in all_group_lines])
    med_s=float(np.median(slopes)) if slopes.size else 0
    mad=float(np.median(np.abs(slopes-med_s))) if slopes.size else 0
    tilt_cap=min(.18,max(.06,med_s+5*mad))
    filtered=[]
    for g in staves:
        gg=[l for l in g if abs(l.get('slope',0))<=tilt_cap]
        if len(gg)>=4: filtered.append(sorted(gg,key=lambda l:l['meanY']))
    staves=sorted(filtered,key=lambda g:np.mean([l['meanY'] for l in g]))
    result={
        'width':bgr.shape[1],'height':bgr.shape[0],
        'metrics':{'spacing':staff_space,'thickness':staff_h,'scaleClass':scale_class,'estStaves':len(staves)},
        'staves':[],
        'debug':{'sauvolaWin':win,'adaptiveThreshold':resp_thr,'adaptiveKernel':klen,'stripCount':N,'blackPeak':bp,'whitePeak':wp,'rawLines':len(lines),'orphans':len(orphans),'tiltCap':tilt_cap,'refLen':ref_len}
    }
    for si,g in enumerate(staves):
        g=sorted(g,key=lambda l:l['meanY'])
        result['staves'].append({'id':si,'lineCount':len(g),'yTop':g[0]['meanY'],'yBottom':g[-1]['meanY'],'x0':min(l['x0'] for l in g),'x1':max(l['x1'] for l in g),'lines':[{'points':l['pts'].tolist(),'meanY':l['meanY'],'slope':l.get('slope',0)} for l in g]})
    return result,gray,bin_img,line_ink

def staff_overlay(bgr,result):
    out=bgr.copy()
    for s in result['staves']:
        for line in s['lines']:
            pts=np.round(np.asarray(line['points'])).astype(np.int32).reshape(-1,1,2)
            cv2.polylines(out,[pts],False,(0,255,0),2,cv2.LINE_AA)
    return out

def interpolate_polyline(points,w):
    pts=np.asarray(points,dtype=float)
    xs=pts[:,0]; ys=pts[:,1]
    slope=ls_slope(pts)
    xx=np.arange(w,dtype=float)
    yy=np.interp(xx,xs,ys)
    left=xx<xs[0]; right=xx>xs[-1]
    yy[left]=ys[0]+slope*(xx[left]-xs[0]); yy[right]=ys[-1]+slope*(xx[right]-xs[-1])
    return yy

def make_cleanplate(gray,bin_img,result):
    h,w=gray.shape
    spacing=result['metrics']['spacing']; thick=result['metrics']['thickness']
    line_mask=np.zeros((h,w),np.uint8)
    for s in result['staves']:
        for line in s['lines']:
            pts=np.round(np.asarray(line['points'])).astype(np.int32).reshape(-1,1,2)
            cv2.polylines(line_mask,[pts],False,255,max(1,round(thick)+2),cv2.LINE_8)
    # Candidate staff pixels: line mask AND local horizontal continuity.
    ink=(bin_img*255).astype(np.uint8)
    hk=max(9,round(spacing*1.6))
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(hk,1)))
    remove=cv2.bitwise_and(line_mask,cv2.dilate(horizontal,cv2.getStructuringElement(cv2.MORPH_RECT,(3,1))))
    # Protect strong vertical strokes and compact notehead cores.
    vk=max(5,round(spacing*1.1))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,vk)))
    protect_v=cv2.dilate(vertical,cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)))
    remove[protect_v>0]=0
    cleaned=ink.copy(); cleaned[remove>0]=0
    # reconnect strokes crossing removed line by a very small vertical close
    cleaned=cv2.morphologyEx(cleaned,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(3,round(thick)+2))))
    return cleaned,line_mask,remove


def robust_quadratic_fit(sequence, clip_px):
    y = np.asarray(sequence, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    valid = np.isfinite(y)
    if int(valid.sum()) < 10:
        return None

    for _ in range(10):
        if int(valid.sum()) < 10:
            return None
        coeff = np.polyfit(x[valid], y[valid], 2)
        residual = np.abs(y - np.polyval(coeff, x))
        updated = np.isfinite(y) & (residual <= clip_px)
        if int(updated.sum()) < max(10, int(0.25 * len(y))):
            break
        if np.array_equal(updated, valid):
            valid = updated
            break
        valid = updated

    coeff = np.polyfit(x[valid], y[valid], 2)
    prediction = np.polyval(coeff, x)
    residual = np.abs(y - prediction)
    inliers = np.isfinite(y) & (residual <= clip_px)
    indices = np.flatnonzero(inliers)
    if indices.size < 10:
        return None

    best_start = int(indices[0])
    best_end = int(indices[0])
    run_start = int(indices[0])
    previous = int(indices[0])

    for index in indices[1:]:
        index = int(index)
        if index - previous > 3:
            if previous - run_start > best_end - best_start:
                best_start = run_start
                best_end = previous
            run_start = index
        previous = index

    if previous - run_start > best_end - best_start:
        best_start = run_start
        best_end = previous

    run = (
        (x >= best_start)
        & (x <= best_end)
        & inliers
    )
    if int(run.sum()) < 10:
        return None

    coeff = np.polyfit(x[run], y[run], 2)
    prediction = np.polyval(coeff, x)
    residual = np.abs(y - prediction)
    final_inliers = (
        np.isfinite(y)
        & (x >= best_start)
        & (x <= best_end)
        & (residual <= clip_px)
    )

    width = best_end - best_start + 1
    coverage = float(final_inliers.sum() / max(1, width))
    rmse = float(np.sqrt(np.mean(residual[final_inliers] ** 2)))
    sag = float(abs(coeff[0]) * max(1, width - 1) ** 2 / 4.0)

    return {
        'coefficients': coeff,
        'x0': best_start,
        'x1': best_end,
        'coverage': coverage,
        'rmse': rmse,
        'sag': sag,
        'inlier_count': int(final_inliers.sum()),
    }


def component_boundary_sequence(component, mode):
    height, width = component.shape
    values = []
    for x in range(width):
        ys = np.flatnonzero(component[:, x])
        if ys.size == 0:
            values.append(np.nan)
        elif mode == 'top':
            values.append(float(ys.min()))
        elif mode == 'bottom':
            values.append(float(ys.max()))
        else:
            values.append(float(np.median(ys)))
    return values


def detect_slurs_and_ties(clean_ink, original_bgr, staff_result):
    spacing = float(staff_result['metrics']['spacing'])
    thickness = float(staff_result['metrics']['thickness'])
    binary = (clean_ink > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    overlay = original_bgr.copy()
    debug_overlay = original_bgr.copy()
    detections = []

    min_width = max(28, int(round(1.7 * spacing)))
    max_height = max(110, int(round(8.0 * spacing)))
    clip_px = max(3.5, 1.15 * thickness)

    for component_id in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[component_id])
        if width < min_width or height < 2 or height > max_height:
            continue
        if width / max(1, height) < 1.45:
            continue
        if area / max(1, width * height) > 0.42:
            continue

        component = labels[y:y + height, x:x + width] == component_id
        candidates = []

        for mode in ('top', 'bottom', 'median'):
            sequence = component_boundary_sequence(component, mode)
            fit = robust_quadratic_fit(sequence, clip_px)
            if fit is None:
                continue

            local_width = fit['x1'] - fit['x0'] + 1
            coefficients = fit['coefficients']
            slope_start = float(2.0 * coefficients[0] * fit['x0'] + coefficients[1])
            slope_end = float(2.0 * coefficients[0] * fit['x1'] + coefficients[1])

            finite_sequence = np.asarray(sequence, dtype=np.float64)
            finite_sequence = finite_sequence[np.isfinite(finite_sequence)]
            steps = np.abs(np.diff(finite_sequence))
            raw_max_step = float(steps.max()) if steps.size else 0.0

            fit.update({
                'mode': mode,
                'width': local_width,
                'slope_start': slope_start,
                'slope_end': slope_end,
                'raw_max_step': raw_max_step,
                'score': local_width * fit['coverage'] - 10.0 * fit['rmse'],
            })

            if local_width < min_width:
                continue
            if fit['coverage'] < 0.55:
                continue
            if fit['rmse'] > max(2.5, 0.9 * thickness):
                continue
            if fit['sag'] < max(2.5, 0.14 * spacing):
                continue
            if fit['sag'] / max(1, local_width) < 0.015:
                continue
            if fit['sag'] > max(120, 8.0 * spacing):
                continue
            if max(abs(slope_start), abs(slope_end)) > 1.1:
                continue
            if slope_start * slope_end > 0 and min(abs(slope_start), abs(slope_end)) > 0.08:
                continue
            if local_width < 5.0 * spacing and raw_max_step > 3.5:
                continue

            candidates.append(fit)

        if not candidates:
            continue

        best = max(candidates, key=lambda item: item['score'])
        coefficients = best['coefficients']
        local_x0 = int(best['x0'])
        local_x1 = int(best['x1'])
        radius = max(2, int(round(thickness * 1.7)))

        selected = np.zeros_like(component, dtype=np.uint8)
        for local_x in range(local_x0, local_x1 + 1):
            local_y = int(round(np.polyval(coefficients, local_x)))
            y0 = max(0, local_y - radius)
            y1 = min(height, local_y + radius + 1)
            selected[y0:y1, local_x] = component[y0:y1, local_x]

        selected_y, selected_x = np.nonzero(selected)
        if selected_x.size < max(20, int(0.5 * (local_x1 - local_x0 + 1))):
            continue

        global_mask = np.zeros(binary.shape, dtype=bool)
        global_mask[y:y + height, x:x + width] = selected.astype(bool)
        overlay[global_mask] = (0, 0, 255)
        debug_overlay[global_mask] = (0, 0, 255)

        sample_count = min(160, local_x1 - local_x0 + 1)
        curve_points = []
        for local_x in np.linspace(local_x0, local_x1, sample_count):
            global_x = int(round(x + local_x))
            global_y = int(round(y + np.polyval(coefficients, local_x)))
            curve_points.append([global_x, global_y])

        curve = np.asarray(curve_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [curve], False, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.polylines(debug_overlay, [curve], False, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(
            debug_overlay,
            (x + local_x0, y),
            (x + local_x1, y + height - 1),
            (0, 0, 255),
            1,
        )

        detections.append({
            'id': len(detections),
            'type': 'slur_or_tie',
            'component_id': component_id,
            'bbox': [x + local_x0, y, local_x1 - local_x0 + 1, height],
            'boundary_mode': best['mode'],
            'coverage': float(best['coverage']),
            'rmse': float(best['rmse']),
            'sag': float(best['sag']),
            'quadratic_local': [float(value) for value in coefficients],
            'curve_points': curve.reshape(-1, 2).tolist(),
        })

    return overlay, debug_overlay, detections



def _staff_band_contains(y, staff_result, margin):
    for stave in staff_result['staves']:
        if stave['yTop'] - margin <= y <= stave['yBottom'] + margin:
            return True
    return False


def _trace_skeleton_chains(binary_mask):
    skeleton = skeletonize(binary_mask > 0)
    degree = convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        mode='constant',
        cval=0,
    ) - skeleton.astype(np.uint8)

    ys, xs = np.nonzero(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    nodes = {(y, x) for y, x in pixels if degree[y, x] != 2}
    offsets = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    ]
    visited_edges = set()
    chains = []

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    def neighbours(point):
        y, x = point
        return [
            (y + dy, x + dx)
            for dy, dx in offsets
            if (y + dy, x + dx) in pixels
        ]

    for node in list(nodes):
        for neighbour in neighbours(node):
            key = edge_key(node, neighbour)
            if key in visited_edges:
                continue
            visited_edges.add(key)
            chain = [node, neighbour]
            previous = node
            current = neighbour

            while current not in nodes:
                next_points = [
                    point
                    for point in neighbours(current)
                    if point != previous
                    and edge_key(current, point) not in visited_edges
                ]
                if not next_points:
                    break
                next_point = next_points[0]
                visited_edges.add(edge_key(current, next_point))
                chain.append(next_point)
                previous, current = current, next_point

            chains.append(chain)

    for point in list(pixels):
        for neighbour in neighbours(point):
            key = edge_key(point, neighbour)
            if key in visited_edges:
                continue
            visited_edges.add(key)
            chain = [point, neighbour]
            previous = point
            current = neighbour

            while True:
                next_points = [
                    candidate
                    for candidate in neighbours(current)
                    if candidate != previous
                    and edge_key(current, candidate) not in visited_edges
                ]
                if not next_points:
                    break
                next_point = next_points[0]
                visited_edges.add(edge_key(current, next_point))
                chain.append(next_point)
                previous, current = current, next_point
                if current == point:
                    break

            chains.append(chain)

    return chains


def _robust_chain_fit(points_xy, clip_px):
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 10:
        return None

    xs = points[:, 0]
    ys = points[:, 1]
    x0 = float(xs.min())
    local_x = xs - x0
    valid = np.ones(len(points), dtype=bool)

    for _ in range(8):
        if int(valid.sum()) < 10:
            return None
        coefficients = np.polyfit(local_x[valid], ys[valid], 2)
        residual = np.abs(ys - np.polyval(coefficients, local_x))
        updated = residual <= clip_px
        if np.array_equal(updated, valid):
            break
        if int(updated.sum()) < 10:
            break
        valid = updated

    if int(valid.sum()) < 10:
        return None

    coefficients = np.polyfit(local_x[valid], ys[valid], 2)
    prediction = np.polyval(coefficients, local_x[valid])
    residual = ys[valid] - prediction
    rmse = float(np.sqrt(np.mean(residual * residual)))
    fit_x0 = float(xs[valid].min())
    fit_x1 = float(xs[valid].max())
    width = fit_x1 - fit_x0 + 1.0
    local_start = fit_x0 - x0
    local_end = fit_x1 - x0
    sag = float(abs(coefficients[0]) * max(1.0, width - 1.0) ** 2 / 4.0)
    slope_start = float(2.0 * coefficients[0] * local_start + coefficients[1])
    slope_end = float(2.0 * coefficients[0] * local_end + coefficients[1])
    unique_x_coverage = float(
        len(np.unique(np.rint(xs[valid]).astype(np.int32))) / max(1.0, width)
    )

    return {
        'origin_x': x0,
        'coefficients': coefficients,
        'x0': fit_x0,
        'x1': fit_x1,
        'width': width,
        'rmse': rmse,
        'sag': sag,
        'slope_start': slope_start,
        'slope_end': slope_end,
        'coverage': unique_x_coverage,
        'inlier_count': int(valid.sum()),
    }


def _curve_y(detection, xs):
    xs = np.asarray(xs, dtype=np.float64)
    if 'quadratic_global_local' in detection:
        origin_x = float(detection['quadratic_global_local']['origin_x'])
        coefficients = np.asarray(
            detection['quadratic_global_local']['coefficients'],
            dtype=np.float64,
        )
        return np.polyval(coefficients, xs - origin_x)

    points = np.asarray(detection['curve_points'], dtype=np.float64)
    order = np.argsort(points[:, 0])
    points = points[order]
    return np.interp(xs, points[:, 0], points[:, 1])


def _detections_duplicate(first, second, spacing):
    first_x0 = float(first['bbox'][0])
    first_x1 = first_x0 + float(first['bbox'][2]) - 1.0
    second_x0 = float(second['bbox'][0])
    second_x1 = second_x0 + float(second['bbox'][2]) - 1.0
    overlap_x0 = max(first_x0, second_x0)
    overlap_x1 = min(first_x1, second_x1)
    overlap = overlap_x1 - overlap_x0 + 1.0
    if overlap <= 0:
        return False

    minimum_width = min(first_x1 - first_x0 + 1.0, second_x1 - second_x0 + 1.0)
    if overlap < 0.45 * minimum_width:
        return False

    samples = np.linspace(overlap_x0, overlap_x1, min(80, max(8, int(overlap))))
    first_y = _curve_y(first, samples)
    second_y = _curve_y(second, samples)
    median_distance = float(np.median(np.abs(first_y - second_y)))
    return median_distance < max(3.0, 0.28 * spacing)


def _curve_vertical_run_stats(original_binary, detection, spacing, thickness):
    x0=float(detection['bbox'][0])
    x1=x0+float(detection['bbox'][2])-1.0
    if x1<=x0:
        return {'median_run':0.0,'thick_fraction':0.0,'sample_count':0}

    samples=np.linspace(x0,x1,min(180,max(24,int(x1-x0+1))))
    predicted=_curve_y(detection,samples)
    search_radius=max(3,int(round(0.45*spacing)))
    thick_threshold=max(6,int(round(1.9*thickness)))
    runs=[]

    h,w=original_binary.shape
    for xf,yf in zip(samples,predicted):
        x=int(round(xf))
        if x<0 or x>=w:
            continue
        y=int(round(yf))
        lo=max(0,y-search_radius)
        hi=min(h,y+search_radius+1)
        ink_positions=np.flatnonzero(original_binary[lo:hi,x]>0)
        if ink_positions.size==0:
            runs.append(0)
            continue
        absolute=ink_positions+lo
        nearest=int(absolute[np.argmin(np.abs(absolute-y))])
        top=nearest
        bottom=nearest
        while top>0 and original_binary[top-1,x]>0:
            top-=1
        while bottom+1<h and original_binary[bottom+1,x]>0:
            bottom+=1
        runs.append(bottom-top+1)

    if not runs:
        return {'median_run':0.0,'thick_fraction':0.0,'sample_count':0}
    positive=[value for value in runs if value>0]
    median_run=float(np.median(positive)) if positive else 0.0
    thick_fraction=float(sum(value>=thick_threshold for value in runs)/len(runs))
    return {
        'median_run':median_run,
        'thick_fraction':thick_fraction,
        'sample_count':len(runs),
        'thick_threshold':thick_threshold,
    }


def _is_probable_beam_edge(original_binary, detection, spacing, thickness):
    stats=_curve_vertical_run_stats(
        original_binary,
        detection,
        spacing,
        thickness,
    )
    width=float(detection['bbox'][2])
    short_or_medium=width <= 8.5*spacing
    reject=(
        short_or_medium
        and stats['median_run'] >= max(5.0,1.65*thickness)
        and stats['thick_fraction'] >= 0.42
    )
    return reject,stats


def detect_skeleton_curve_chains(clean_ink, original_binary, staff_result):
    spacing = float(staff_result['metrics']['spacing'])
    thickness = float(staff_result['metrics']['thickness'])
    margin = 4.5 * spacing
    min_width = max(28.0, 1.7 * spacing)
    clip_px = max(2.0, 0.8 * thickness)
    chains = _trace_skeleton_chains(clean_ink > 0)
    candidates = []

    for chain_id, chain in enumerate(chains):
        if len(chain) < 10:
            continue

        points = np.asarray([(x, y) for y, x in chain], dtype=np.float64)
        xs = points[:, 0]
        ys = points[:, 1]
        x_span = float(xs.max() - xs.min() + 1.0)
        y_span = float(ys.max() - ys.min() + 1.0)
        mean_y = float(np.mean(ys))

        if not _staff_band_contains(mean_y, staff_result, margin):
            continue
        if x_span < min_width:
            continue
        if y_span > max(90.0, 6.0 * spacing):
            continue
        if x_span / max(1.0, y_span) < 1.45:
            continue

        fit = _robust_chain_fit(points, clip_px)
        if fit is None:
            continue

        chain_length = float(
            sum(
                np.linalg.norm(points[index] - points[index - 1])
                for index in range(1, len(points))
            )
        )
        length_ratio = chain_length / max(1.0, fit['width'])

        if fit['width'] < min_width:
            continue
        if fit['coverage'] < 0.55:
            continue
        if fit['rmse'] > max(1.8, 0.75 * thickness):
            continue
        if fit['sag'] < max(2.2, 0.12 * spacing):
            continue
        if fit['sag'] / max(1.0, fit['width']) < 0.014:
            continue
        if fit['sag'] > max(130.0, 8.0 * spacing):
            continue
        if max(abs(fit['slope_start']), abs(fit['slope_end'])) > 1.15:
            continue
        if (
            fit['slope_start'] * fit['slope_end'] > 0
            and min(abs(fit['slope_start']), abs(fit['slope_end'])) > 0.12
        ):
            continue
        if length_ratio > 2.25:
            continue

        sample_count = min(240, max(40, int(round(fit['width']))))
        sample_x = np.linspace(fit['x0'], fit['x1'], sample_count)
        sample_y = np.polyval(
            fit['coefficients'],
            sample_x - fit['origin_x'],
        )
        curve_points = np.column_stack([sample_x, sample_y])
        bbox_y0 = int(np.floor(sample_y.min() - 2.0 * thickness))
        bbox_y1 = int(np.ceil(sample_y.max() + 2.0 * thickness))

        candidate={
            'id': -1,
            'type': 'slur_or_tie',
            'source': 'skeleton_chain',
            'chain_id': chain_id,
            'bbox': [
                int(round(fit['x0'])),
                bbox_y0,
                int(round(fit['width'])),
                max(1, bbox_y1 - bbox_y0 + 1),
            ],
            'coverage': float(fit['coverage']),
            'rmse': float(fit['rmse']),
            'sag': float(fit['sag']),
            'slope_start': float(fit['slope_start']),
            'slope_end': float(fit['slope_end']),
            'quadratic_global_local': {
                'origin_x': float(fit['origin_x']),
                'coefficients': [float(value) for value in fit['coefficients']],
            },
            'curve_points': np.rint(curve_points).astype(np.int32).tolist(),
            'score': float(
                fit['width']
                * fit['coverage']
                + 5.0 * fit['sag']
                - 12.0 * fit['rmse']
            ),
        }
        beam_reject,beam_stats=_is_probable_beam_edge(
            original_binary,
            candidate,
            spacing,
            thickness,
        )
        candidate['beam_profile']=beam_stats
        if beam_reject:
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda item: item['score'], reverse=True)
    kept = []
    for candidate in candidates:
        if any(_detections_duplicate(candidate, other, spacing) for other in kept):
            continue
        kept.append(candidate)

    return kept


def combine_curve_detections(base_detections, skeleton_detections, original_binary, staff_result):
    spacing = float(staff_result['metrics']['spacing'])
    combined = []

    for detection in base_detections:
        item = dict(detection)
        item['source'] = 'component_boundary'
        item['score'] = float(
            item['bbox'][2] * item.get('coverage', 0.0)
            + 5.0 * item.get('sag', 0.0)
            - 12.0 * item.get('rmse', 0.0)
        )
        beam_reject,beam_stats=_is_probable_beam_edge(
            original_binary,
            item,
            spacing,
            float(staff_result['metrics']['thickness']),
        )
        item['beam_profile']=beam_stats
        if beam_reject:
            continue
        combined.append(item)

    for detection in skeleton_detections:
        if any(_detections_duplicate(detection, other, spacing) for other in combined):
            continue
        combined.append(detection)

    combined.sort(key=lambda item: item.get('score', 0.0), reverse=True)
    final = []
    for detection in combined:
        if any(_detections_duplicate(detection, other, spacing) for other in final):
            continue
        detection = dict(detection)
        detection['id'] = len(final)
        final.append(detection)

    final.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    for index, detection in enumerate(final):
        detection['id'] = index
    return final





def _detection_x_range(detection):
    points = np.asarray(detection['curve_points'], dtype=np.float64)
    return float(points[:, 0].min()), float(points[:, 0].max())


def _curve_slope(detection, xs):
    xs = np.asarray(xs, dtype=np.float64)
    if 'quadratic_global_local' in detection:
        model = detection['quadratic_global_local']
        origin_x = float(model['origin_x'])
        coefficients = np.asarray(model['coefficients'], dtype=np.float64)
        local_x = xs - origin_x
        if len(coefficients) == 3:
            return 2.0 * coefficients[0] * local_x + coefficients[1]
        return np.full_like(xs, coefficients[0], dtype=np.float64)

    points = np.asarray(detection['curve_points'], dtype=np.float64)
    points = points[np.argsort(points[:, 0])]
    if len(points) < 3:
        slope = (points[-1, 1] - points[0, 1]) / max(1.0, points[-1, 0] - points[0, 0])
        return np.full_like(xs, slope, dtype=np.float64)
    gradients = np.gradient(points[:, 1], points[:, 0])
    return np.interp(xs, points[:, 0], gradients)


def _fit_merged_detection_model(detections, thickness):
    points = np.vstack([
        np.asarray(item['curve_points'], dtype=np.float64)
        for item in detections
    ])
    points = points[np.argsort(points[:, 0])]
    x_origin = float(points[:, 0].min())
    local_x = points[:, 0] - x_origin
    valid = np.ones(len(points), dtype=bool)
    clip = max(1.8, 0.9 * thickness)

    for _ in range(8):
        if int(valid.sum()) < 12:
            return None
        coefficients = np.polyfit(local_x[valid], points[valid, 1], 2)
        residual = np.abs(points[:, 1] - np.polyval(coefficients, local_x))
        updated = residual <= clip
        if int(updated.sum()) < 12:
            return None
        if np.array_equal(updated, valid):
            valid = updated
            break
        valid = updated

    coefficients = np.polyfit(local_x[valid], points[valid, 1], 2)
    residual = np.abs(points[:, 1] - np.polyval(coefficients, local_x))
    return {
        'origin_x': x_origin,
        'coefficients': coefficients,
        'median_residual': float(np.median(residual[valid])),
        'max_residual_p90': float(np.percentile(residual[valid], 90)),
    }


def _corridor_support_stats(
    original_binary,
    staff_mask,
    removed_staff_pixels,
    model,
    x0,
    x1,
    spacing,
    thickness,
):
    if x1 < x0:
        return {
            'coverage': 1.0,
            'staff_fraction': 0.0,
            'thin_fraction': 1.0,
            'count': 0,
        }

    height, width = original_binary.shape
    x0 = max(0, int(np.floor(x0)))
    x1 = min(width - 1, int(np.ceil(x1)))
    if x1 < x0:
        return {
            'coverage': 0.0,
            'staff_fraction': 0.0,
            'thin_fraction': 0.0,
            'count': 0,
        }

    radius = max(2, int(round(1.15 * thickness)))
    max_thin_run = max(6, int(round(2.4 * thickness)))
    supported = 0
    staff_hits = 0
    thin_hits = 0
    count = 0

    for x in range(x0, x1 + 1):
        local_x = float(x) - float(model['origin_x'])
        predicted_y = float(np.polyval(model['coefficients'], local_x))
        center_y = int(round(predicted_y))
        y0 = max(0, center_y - radius)
        y1 = min(height, center_y + radius + 1)
        positions = np.flatnonzero(original_binary[y0:y1, x] > 0)
        count += 1
        if positions.size == 0:
            continue

        absolute = positions + y0
        y = int(absolute[np.argmin(np.abs(absolute.astype(np.float64) - predicted_y))])
        run = _vertical_run_length(original_binary, x, y)
        staff_overlap = bool(
            np.any(staff_mask[y0:y1, x] > 0)
            or np.any(removed_staff_pixels[y0:y1, x] > 0)
        )
        supported += 1
        staff_hits += int(staff_overlap)
        thin_hits += int(run <= max_thin_run)

    return {
        'coverage': float(supported / max(1, count)),
        'staff_fraction': float(staff_hits / max(1, count)),
        'thin_fraction': float(thin_hits / max(1, supported)),
        'count': int(count),
    }


def _can_merge_curve_fragments(
    first,
    second,
    original_binary,
    staff_mask,
    removed_staff_pixels,
    spacing,
    thickness,
):
    first_x0, first_x1 = _detection_x_range(first)
    second_x0, second_x1 = _detection_x_range(second)

    if second_x0 < first_x0:
        first, second = second, first
        first_x0, first_x1, second_x0, second_x1 = (
            second_x0,
            second_x1,
            first_x0,
            first_x1,
        )

    overlap = first_x1 - second_x0 + 1.0
    if overlap > 0.35 * min(first_x1 - first_x0 + 1.0, second_x1 - second_x0 + 1.0):
        samples = np.linspace(
            second_x0,
            min(first_x1, second_x1),
            max(8, int(min(first_x1, second_x1) - second_x0 + 1.0)),
        )
        separation = float(np.median(np.abs(_curve_y(first, samples) - _curve_y(second, samples))))
        return separation < max(2.0, 0.22 * spacing), 1000.0 - separation

    gap = second_x0 - first_x1 - 1.0
    if gap < -0.2 * spacing or gap > 4.0 * spacing:
        return False, -1e9

    join_x = np.asarray([first_x1, second_x0], dtype=np.float64)
    join_y_first = float(_curve_y(first, [first_x1])[0])
    join_y_second = float(_curve_y(second, [second_x0])[0])
    if abs(join_y_first - join_y_second) > max(8.0, 0.65 * spacing):
        return False, -1e9

    slope_first = float(_curve_slope(first, [first_x1])[0])
    slope_second = float(_curve_slope(second, [second_x0])[0])
    if abs(slope_first - slope_second) > 0.34:
        return False, -1e9

    model = _fit_merged_detection_model([first, second], thickness)
    if model is None:
        return False, -1e9
    if model['max_residual_p90'] > max(2.5, 1.15 * thickness):
        return False, -1e9

    support = _corridor_support_stats(
        original_binary,
        staff_mask,
        removed_staff_pixels,
        model,
        first_x1 + 1,
        second_x0 - 1,
        spacing,
        thickness,
    )
    if gap > 2.0:
        enough_observed_ink = (
            support['coverage'] >= 0.28
            and support['thin_fraction'] >= 0.45
        )
        staff_interruption = (
            support['coverage'] >= 0.12
            and support['staff_fraction'] >= 0.18
        )
        if not (enough_observed_ink or staff_interruption):
            return False, -1e9

    score = (
        120.0
        - 8.0 * gap
        - 25.0 * abs(join_y_first - join_y_second)
        - 80.0 * abs(slope_first - slope_second)
        - 20.0 * model['median_residual']
        + 60.0 * support['coverage']
        + 20.0 * support['staff_fraction']
    )
    return True, float(score)


def _merge_detection_cluster(cluster, thickness):
    model = _fit_merged_detection_model(cluster, thickness)
    if model is None:
        return None

    x0 = min(_detection_x_range(item)[0] for item in cluster)
    x1 = max(_detection_x_range(item)[1] for item in cluster)
    sample_count = min(320, max(60, int(round(x1 - x0 + 1.0))))
    sample_x = np.linspace(x0, x1, sample_count)
    sample_y = np.polyval(model['coefficients'], sample_x - model['origin_x'])
    points = np.column_stack([sample_x, sample_y])
    y0 = int(np.floor(sample_y.min() - 2.5 * thickness))
    y1 = int(np.ceil(sample_y.max() + 2.5 * thickness))

    merged = {
        'id': -1,
        'type': 'slur_or_tie',
        'source': 'merged_fragments',
        'fragment_ids': [int(item.get('id', -1)) for item in cluster],
        'bbox': [
            int(np.floor(x0)),
            y0,
            max(1, int(np.ceil(x1)) - int(np.floor(x0)) + 1),
            max(1, y1 - y0 + 1),
        ],
        'coverage': float(np.mean([item.get('coverage', 0.0) for item in cluster])),
        'rmse': float(model['median_residual']),
        'sag': float(
            abs(model['coefficients'][0])
            * max(1.0, x1 - x0) ** 2
            / 4.0
        ),
        'quadratic_global_local': {
            'origin_x': float(model['origin_x']),
            'coefficients': [float(value) for value in model['coefficients']],
        },
        'curve_points': np.rint(points).astype(np.int32).tolist(),
        'score': float(sum(item.get('score', 0.0) for item in cluster)),
    }
    return merged


def merge_curve_fragments(
    detections,
    original_binary,
    staff_mask,
    removed_staff_pixels,
    staff_result,
):
    spacing = float(staff_result['metrics']['spacing'])
    thickness = float(staff_result['metrics']['thickness'])
    clusters = [[dict(item)] for item in detections]

    while True:
        best_pair = None
        best_score = -1e9
        best_merged = None

        for first_index in range(len(clusters)):
            first_detection = _merge_detection_cluster(clusters[first_index], thickness)
            if first_detection is None:
                continue
            for second_index in range(first_index + 1, len(clusters)):
                second_detection = _merge_detection_cluster(clusters[second_index], thickness)
                if second_detection is None:
                    continue
                can_merge, score = _can_merge_curve_fragments(
                    first_detection,
                    second_detection,
                    original_binary,
                    staff_mask,
                    removed_staff_pixels,
                    spacing,
                    thickness,
                )
                if can_merge and score > best_score:
                    merged = _merge_detection_cluster(
                        clusters[first_index] + clusters[second_index],
                        thickness,
                    )
                    if merged is not None:
                        best_pair = (first_index, second_index)
                        best_score = score
                        best_merged = merged

        if best_pair is None:
            break

        first_index, second_index = best_pair
        combined_cluster = clusters[first_index] + clusters[second_index]
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in best_pair
        ]
        clusters.append(combined_cluster)

    merged_detections = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged = dict(cluster[0])
        else:
            merged = _merge_detection_cluster(cluster, thickness)
            if merged is None:
                merged_detections.extend(cluster)
                continue
        merged_detections.append(merged)

    merged_detections.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    for index, item in enumerate(merged_detections):
        item['id'] = index
    return merged_detections


def _fit_curve_extension_model(curve_points):
    points = np.asarray(curve_points, dtype=np.float64)
    points = points[np.argsort(points[:, 0])]

    xs = points[:, 0]
    ys = points[:, 1]
    origin_x = float(xs.min())
    local_x = xs - origin_x
    valid = np.ones(len(points), dtype=bool)

    for _ in range(6):
        degree = 2 if int(valid.sum()) >= 6 else 1
        coefficients = np.polyfit(
            local_x[valid],
            ys[valid],
            degree,
        )
        prediction = np.polyval(coefficients, local_x)
        residual = np.abs(ys - prediction)

        median_residual = float(np.median(residual[valid]))
        mad = float(
            np.median(
                np.abs(residual[valid] - median_residual)
            )
        )
        clip_px = max(
            1.5,
            median_residual + 3.0 * max(0.5, mad),
        )
        updated = residual <= clip_px

        if np.array_equal(updated, valid):
            break
        if int(updated.sum()) < max(6, int(0.5 * len(points))):
            break
        valid = updated

    degree = 2 if int(valid.sum()) >= 6 else 1
    coefficients = np.polyfit(
        local_x[valid],
        ys[valid],
        degree,
    )

    return {
        'points': points,
        'origin_x': origin_x,
        'coefficients': coefficients,
    }


def _curve_model_y(model, x):
    return float(
        np.polyval(
            model['coefficients'],
            float(x) - model['origin_x'],
        )
    )


def _curve_model_slope(model, x):
    coefficients = model['coefficients']

    if len(coefficients) == 3:
        local_x = float(x) - model['origin_x']
        return float(
            2.0 * coefficients[0] * local_x
            + coefficients[1]
        )

    return float(coefficients[0])


def _vertical_run_near_curve(
    original_binary,
    x,
    predicted_y,
    search_radius,
):
    height, width = original_binary.shape

    if x < 0 or x >= width:
        return None

    center_y = int(round(predicted_y))
    y0 = max(0, center_y - search_radius)
    y1 = min(height, center_y + search_radius + 1)
    positions = np.flatnonzero(
        original_binary[y0:y1, x] > 0
    )

    if positions.size == 0:
        return None

    absolute_positions = positions + y0
    nearest_y = int(
        absolute_positions[
            np.argmin(
                np.abs(
                    absolute_positions.astype(np.float64)
                    - predicted_y
                )
            )
        ]
    )

    top = nearest_y
    bottom = nearest_y

    while (
        top > 0
        and original_binary[top - 1, x] > 0
    ):
        top -= 1

    while (
        bottom + 1 < height
        and original_binary[bottom + 1, x] > 0
    ):
        bottom += 1

    return {
        'nearest_y': nearest_y,
        'deviation': float(nearest_y - predicted_y),
        'run': int(bottom - top + 1),
    }


def _trajectory_support_profile(
    detection,
    side,
    original_binary,
    removed_staff_pixels,
    spacing,
    thickness,
):
    model = _fit_curve_extension_model(
        detection['curve_points']
    )
    points = model['points']

    endpoint = (
        points[0]
        if side == 'left'
        else points[-1]
    )
    endpoint_x = int(round(endpoint[0]))
    direction = -1 if side == 'left' else 1
    max_extension = max(
        24,
        int(round(8.0 * spacing)),
    )

    xs = np.arange(
        endpoint_x + direction,
        endpoint_x + direction * (max_extension + 1),
        direction,
        dtype=np.int32,
    )

    height, width = original_binary.shape
    xs = xs[(xs >= 0) & (xs < width)]

    corridor_radius = max(
        2,
        int(round(0.85 * thickness)),
    )
    local_half_window = max(
        5,
        int(round(0.45 * spacing)),
    )
    staff_mask_radius = max(
        2,
        int(round(thickness + 1.0)),
    )
    max_symbol_run = max(
        6,
        int(round(2.0 * thickness)),
    )

    profile = []

    for x in xs:
        predicted_y = _curve_model_y(model, x)
        hits = 0
        thick_hits = 0
        samples = 0

        window_x0 = max(0, x - local_half_window)
        window_x1 = min(
            width,
            x + local_half_window + 1,
        )

        for sample_x in range(window_x0, window_x1):
            sample_y = _curve_model_y(
                model,
                sample_x,
            )
            run = _vertical_run_near_curve(
                original_binary,
                sample_x,
                sample_y,
                corridor_radius,
            )

            if run is not None:
                hits += 1
                if run['run'] > max_symbol_run:
                    thick_hits += 1

            samples += 1

        center_y = int(round(predicted_y))
        staff_y0 = max(
            0,
            center_y - staff_mask_radius,
        )
        staff_y1 = min(
            height,
            center_y + staff_mask_radius + 1,
        )

        removed_support = bool(
            np.any(
                removed_staff_pixels[
                    staff_y0:staff_y1,
                    x,
                ] > 0
            )
        )
        coverage = float(
            hits / max(1, samples)
        )
        thick_fraction = float(
            thick_hits / max(1, samples)
        )

        profile.append({
            'x': int(x),
            'y': float(predicted_y),
            'coverage': coverage,
            'thick_fraction': thick_fraction,
            'removed_support': removed_support,
            'accepted_support': (
                coverage >= 0.48
                and thick_fraction <= 0.42
            ),
        })

    return model, profile


def _extend_one_curve_endpoint_locked(
    detection,
    side,
    original_binary,
    removed_staff_pixels,
    spacing,
    thickness,
):
    model, profile = _trajectory_support_profile(
        detection,
        side,
        original_binary,
        removed_staff_pixels,
        spacing,
        thickness,
    )

    if not profile:
        return None, {
            'reason': 'empty_profile',
        }

    points = model['points']
    endpoint_x = (
        float(points[0, 0])
        if side == 'left'
        else float(points[-1, 0])
    )
    endpoint_slope = _curve_model_slope(
        model,
        endpoint_x,
    )

    if abs(endpoint_slope) < 0.055:
        return None, {
            'reason': 'flat_endpoint',
            'endpoint_slope': endpoint_slope,
        }

    maximum_removed_start_distance = max(
        4,
        int(round(0.85 * spacing)),
    )
    removed_near_endpoint = any(
        item['removed_support']
        for item in profile[
            :maximum_removed_start_distance + 1
        ]
    )

    if not removed_near_endpoint:
        return None, {
            'reason': 'not_staff_deleted',
            'endpoint_slope': endpoint_slope,
        }

    maximum_support_gap = max(
        5,
        int(round(0.58 * spacing)),
    )
    last_supported_index = -1
    unsupported_run = 0

    for index, item in enumerate(profile):
        if item['accepted_support']:
            last_supported_index = index
            unsupported_run = 0
        else:
            unsupported_run += 1
            if unsupported_run > maximum_support_gap:
                break

    minimum_extension = max(
        7,
        int(round(0.48 * spacing)),
    )

    if last_supported_index + 1 < minimum_extension:
        return None, {
            'reason': 'short_support',
            'length': last_supported_index + 1,
            'endpoint_slope': endpoint_slope,
        }

    selected_profile = profile[
        :last_supported_index + 1
    ]
    removed_fraction = float(
        sum(
            item['removed_support']
            for item in selected_profile
        )
        / len(selected_profile)
    )

    if removed_fraction < 0.18:
        return None, {
            'reason': 'weak_staff_overlap',
            'length': len(selected_profile),
            'removed_fraction': removed_fraction,
            'endpoint_slope': endpoint_slope,
        }

    extension_points = np.asarray(
        [
            [item['x'], item['y']]
            for item in selected_profile
        ],
        dtype=np.float64,
    )
    extension_points = extension_points[
        np.argsort(extension_points[:, 0])
    ]

    return extension_points, {
        'reason': 'accepted',
        'length': int(len(extension_points)),
        'removed_fraction': removed_fraction,
        'endpoint_slope': endpoint_slope,
        'mean_coverage': float(
            np.mean(
                [
                    item['coverage']
                    for item in selected_profile
                ]
            )
        ),
    }


def extend_detections_through_staff(
    detections,
    original_binary,
    removed_staff_pixels,
    staff_result,
):
    spacing = float(
        staff_result['metrics']['spacing']
    )
    thickness = float(
        staff_result['metrics']['thickness']
    )
    extended = []

    for detection in detections:
        item = dict(detection)
        points = np.asarray(
            item['curve_points'],
            dtype=np.float64,
        )
        points = points[
            np.argsort(points[:, 0])
        ]

        left_extension, left_debug = (
            _extend_one_curve_endpoint_locked(
                item,
                'left',
                original_binary,
                removed_staff_pixels,
                spacing,
                thickness,
            )
        )
        right_extension, right_debug = (
            _extend_one_curve_endpoint_locked(
                item,
                'right',
                original_binary,
                removed_staff_pixels,
                spacing,
                thickness,
            )
        )

        merged_parts = []

        if left_extension is not None:
            merged_parts.append(left_extension)

        merged_parts.append(points)

        if right_extension is not None:
            merged_parts.append(right_extension)

        merged = np.vstack(merged_parts)
        merged = merged[
            np.argsort(merged[:, 0])
        ]

        rounded_x = np.rint(
            merged[:, 0]
        ).astype(np.int32)
        unique_points = []

        for x in np.unique(rounded_x):
            ys = merged[
                rounded_x == x,
                1,
            ]
            unique_points.append(
                [
                    int(x),
                    float(np.median(ys)),
                ]
            )

        merged = np.asarray(
            unique_points,
            dtype=np.float64,
        )

        item['curve_points'] = np.rint(
            merged
        ).astype(np.int32).tolist()

        x0 = int(np.floor(merged[:, 0].min()))
        x1 = int(np.ceil(merged[:, 0].max()))
        y0 = int(
            np.floor(
                merged[:, 1].min()
                - 2.0 * thickness
            )
        )
        y1 = int(
            np.ceil(
                merged[:, 1].max()
                + 2.0 * thickness
            )
        )

        item['bbox'] = [
            x0,
            y0,
            max(1, x1 - x0 + 1),
            max(1, y1 - y0 + 1),
        ]
        item['staff_crossing_extension'] = {
            'left': (
                0
                if left_extension is None
                else int(
                    round(
                        points[0, 0]
                        - left_extension[:, 0].min()
                    )
                )
            ),
            'right': (
                0
                if right_extension is None
                else int(
                    round(
                        right_extension[:, 0].max()
                        - points[-1, 0]
                    )
                )
            ),
        }
        item['extension_debug'] = {
            'left': left_debug,
            'right': right_debug,
        }

        extended.append(item)

    return extended




def _nearest_mask_y(mask, x, predicted_y, radius):
    height, width = mask.shape
    if x < 0 or x >= width:
        return None
    center = int(round(predicted_y))
    y0 = max(0, center - radius)
    y1 = min(height, center + radius + 1)
    positions = np.flatnonzero(mask[y0:y1, x] > 0)
    if positions.size == 0:
        return None
    absolute = positions + y0
    return int(absolute[np.argmin(np.abs(absolute.astype(np.float64) - predicted_y))])


def _candidate_mask_ys(mask, x, predicted_y, radius):
    height, width = mask.shape
    if x < 0 or x >= width:
        return []
    center = int(round(predicted_y))
    y0 = max(0, center - radius)
    y1 = min(height, center + radius + 1)
    positions = np.flatnonzero(mask[y0:y1, x] > 0)
    return (positions + y0).astype(np.int32).tolist()


def _has_branch_near(branch_points_xy, x, y, radius):
    if branch_points_xy.size == 0:
        return False
    dx = branch_points_xy[:, 0] - float(x)
    dy = branch_points_xy[:, 1] - float(y)
    return bool(np.min(dx * dx + dy * dy) <= float(radius * radius))


def _vertical_run_length(binary_mask, x, y):
    height, width = binary_mask.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return 0
    if binary_mask[y, x] == 0:
        return 0
    top = y
    bottom = y
    while top > 0 and binary_mask[top - 1, x] > 0:
        top -= 1
    while bottom + 1 < height and binary_mask[bottom + 1, x] > 0:
        bottom += 1
    return int(bottom - top + 1)


def _horizontal_run_length(binary_mask, x, y, limit):
    height, width = binary_mask.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return 0
    if binary_mask[y, x] == 0:
        return 0
    left = x
    right = x
    while left > 0 and x - left < limit and binary_mask[y, left - 1] > 0:
        left -= 1
    while right + 1 < width and right - x < limit and binary_mask[y, right + 1] > 0:
        right += 1
    return int(right - left + 1)


def _staff_overlap_at(staff_mask, removed_staff_pixels, x, y, radius):
    height, width = staff_mask.shape
    if x < 0 or x >= width:
        return False
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height, int(round(y)) + radius + 1)
    return bool(
        np.any(staff_mask[y0:y1, x] > 0)
        or np.any(removed_staff_pixels[y0:y1, x] > 0)
    )


def _local_curve_support(original_binary, detection, x, radius_x, radius_y):
    height, width = original_binary.shape
    x0 = max(0, int(x) - radius_x)
    x1 = min(width - 1, int(x) + radius_x)
    if x1 < x0:
        return 0.0
    xs = np.arange(x0, x1 + 1, dtype=np.int32)
    predicted = _curve_y(detection, xs)
    hits = 0
    for sample_x, sample_y in zip(xs, predicted):
        center = int(round(sample_y))
        y0 = max(0, center - radius_y)
        y1 = min(height, center + radius_y + 1)
        hits += int(np.any(original_binary[y0:y1, sample_x] > 0))
    return float(hits / max(1, len(xs)))


def _trace_curve_columns(
    detection,
    clean_skeleton,
    original_binary,
    staff_mask,
    removed_staff_pixels,
    branch_points_xy,
    spacing,
    thickness,
):
    core_x0, core_x1 = _detection_x_range(detection)
    margin = max(4, int(round(1.0 * spacing)))
    x_start = max(0, int(np.floor(core_x0)) - margin)
    x_end = min(original_binary.shape[1] - 1, int(np.ceil(core_x1)) + margin)
    if x_end <= x_start:
        return None

    xs = np.arange(x_start, x_end + 1, dtype=np.int32)
    predicted = _curve_y(detection, xs)
    slopes = _curve_slope(detection, xs)
    clean_radius = max(2, int(round(1.25 * thickness)))
    original_radius = max(clean_radius + 1, int(round(0.24 * spacing)))
    staff_radius = max(2, int(round(1.2 * thickness)))
    maximum_run = max(6, int(round(2.5 * thickness)))
    long_horizontal = max(20, int(round(3.0 * spacing)))
    tangent_radius_x = max(4, int(round(0.35 * spacing)))
    tangent_radius_y = max(2, int(round(1.1 * thickness)))

    traced_y = np.full(len(xs), np.nan, dtype=np.float64)
    source = np.zeros(len(xs), dtype=np.uint8)
    risky = np.zeros(len(xs), dtype=np.uint8)
    previous_index = -1000000
    previous_y = 0.0

    for index, (x, predicted_y, slope) in enumerate(zip(xs, predicted, slopes)):
        candidates = []
        for y in _candidate_mask_ys(clean_skeleton, int(x), float(predicted_y), clean_radius):
            score = 1.8 * abs(float(y) - float(predicted_y))
            candidates.append((score, int(y), 1, 0))

        for y in _candidate_mask_ys(original_binary, int(x), float(predicted_y), original_radius):
            if clean_skeleton[y, x] > 0:
                continue
            staff_overlap = _staff_overlap_at(
                staff_mask,
                removed_staff_pixels,
                int(x),
                int(y),
                staff_radius,
            )
            vertical_run = _vertical_run_length(original_binary, int(x), int(y))
            horizontal_run = _horizontal_run_length(
                original_binary,
                int(x),
                int(y),
                long_horizontal + 1,
            )
            tangent_support = _local_curve_support(
                original_binary,
                detection,
                int(x),
                tangent_radius_x,
                tangent_radius_y,
            )

            if vertical_run > maximum_run and not staff_overlap:
                continue
            if tangent_support < 0.34 and not staff_overlap:
                continue

            horizontal_risk = (
                horizontal_run >= long_horizontal
                and abs(float(slope)) < 0.10
            )
            score = (
                2.2 * abs(float(y) - float(predicted_y))
                + 0.7
                + 1.8 * max(0.0, 0.48 - tangent_support)
                + (2.5 if horizontal_risk else 0.0)
            )
            candidates.append((score, int(y), 2, int(horizontal_risk)))

        if not candidates:
            continue

        if previous_index >= 0 and index - previous_index <= max(3, int(round(0.28 * spacing))):
            expected_delta = float(predicted_y - predicted[previous_index])
            rescored = []
            for score, y, candidate_source, candidate_risk in candidates:
                actual_delta = float(y) - previous_y
                continuity_penalty = 2.4 * abs(actual_delta - expected_delta)
                rescored.append((score + continuity_penalty, y, candidate_source, candidate_risk))
            candidates = rescored

        score, y, candidate_source, candidate_risk = min(candidates, key=lambda item: item[0])
        traced_y[index] = float(y)
        source[index] = int(candidate_source)
        risky[index] = int(candidate_risk)
        previous_index = index
        previous_y = float(y)

    clean_indices = np.flatnonzero(source == 1)
    if clean_indices.size == 0:
        return None

    previous_clean = np.full(len(xs), -1000000, dtype=np.int32)
    next_clean = np.full(len(xs), 1000000, dtype=np.int32)
    last = -1000000
    for index in range(len(xs)):
        if source[index] == 1:
            last = index
        previous_clean[index] = last
    last = 1000000
    for index in range(len(xs) - 1, -1, -1):
        if source[index] == 1:
            last = index
        next_clean[index] = last

    bridge_limit = max(5, int(round(2.4 * spacing)))
    one_sided_limit = max(4, int(round(0.75 * spacing)))
    staff_radius = max(2, int(round(1.2 * thickness)))

    for index in np.flatnonzero(source == 2):
        x = int(xs[index])
        y = int(round(traced_y[index]))
        staff_overlap = _staff_overlap_at(
            staff_mask,
            removed_staff_pixels,
            x,
            y,
            staff_radius,
        )
        left_distance = index - previous_clean[index]
        right_distance = next_clean[index] - index
        bounded_bridge = left_distance <= bridge_limit and right_distance <= bridge_limit
        one_sided = min(left_distance, right_distance) <= one_sided_limit
        thin_run = _vertical_run_length(original_binary, x, y) <= max(6, int(round(2.4 * thickness)))
        local_support = _local_curve_support(
            original_binary,
            detection,
            x,
            max(4, int(round(0.35 * spacing))),
            max(2, int(round(1.1 * thickness))),
        )

        keep = (
            (staff_overlap and bounded_bridge)
            or (
                thin_run
                and one_sided
                and local_support >= 0.52
                and risky[index] == 0
            )
        )
        if not keep:
            source[index] = 0
            traced_y[index] = np.nan

    residual = np.abs(traced_y - predicted)
    residual_limit = max(2.0, 1.2 * thickness)
    invalid = (source > 0) & (residual > residual_limit)
    source[invalid] = 0
    traced_y[invalid] = np.nan

    valid_indices = np.flatnonzero(source > 0)
    if valid_indices.size == 0:
        return None

    max_column_hole = 2
    raw_segments = []
    segment_start = 0
    for index in range(1, len(valid_indices)):
        if valid_indices[index] - valid_indices[index - 1] > max_column_hole + 1:
            raw_segments.append(valid_indices[segment_start:index])
            segment_start = index
    raw_segments.append(valid_indices[segment_start:])

    minimum_segment_points = max(5, int(round(0.24 * spacing)))
    segments = []
    for segment_indices in raw_segments:
        if len(segment_indices) < minimum_segment_points:
            continue
        segment_x = xs[segment_indices].astype(np.float64)
        segment_y = traced_y[segment_indices].astype(np.float64)
        segment_source = source[segment_indices]
        segment_width = float(segment_x[-1] - segment_x[0] + 1.0)
        coverage = float(len(segment_x) / max(1.0, segment_width))
        if coverage < 0.45:
            continue
        segments.append({
            'indices': segment_indices,
            'x': segment_x,
            'y': segment_y,
            'source': segment_source,
            'x0': float(segment_x[0]),
            'x1': float(segment_x[-1]),
            'coverage': coverage,
        })

    if not segments:
        return None

    core_segments = [
        index
        for index, segment in enumerate(segments)
        if segment['x1'] >= core_x0 and segment['x0'] <= core_x1
    ]
    if not core_segments:
        core_segments = [
            int(np.argmax([len(segment['x']) for segment in segments]))
        ]

    selected = set(core_segments)
    maximum_internal_gap = max(6, int(round(2.6 * spacing)))
    maximum_endpoint_gap = max(3, int(round(0.55 * spacing)))
    branch_radius = max(4.0, 0.55 * spacing)

    changed = True
    while changed:
        changed = False
        selected_x0 = min(segments[index]['x0'] for index in selected)
        selected_x1 = max(segments[index]['x1'] for index in selected)

        for index, segment in enumerate(segments):
            if index in selected:
                continue
            if segment['x1'] < selected_x0:
                gap = selected_x0 - segment['x1'] - 1.0
                join_x = segment['x1']
                join_y = segment['y'][-1]
                outside_core = segment['x1'] < core_x0
            elif segment['x0'] > selected_x1:
                gap = segment['x0'] - selected_x1 - 1.0
                join_x = segment['x0']
                join_y = segment['y'][0]
                outside_core = segment['x0'] > core_x1
            else:
                gap = 0.0
                join_x = segment['x0']
                join_y = segment['y'][0]
                outside_core = False

            gap_limit = maximum_endpoint_gap if outside_core else maximum_internal_gap
            if gap > gap_limit:
                continue
            if outside_core and _has_branch_near(branch_points_xy, join_x, join_y, branch_radius):
                continue

            gap_x0 = min(selected_x1, segment['x1']) + 1.0
            gap_x1 = max(selected_x0, segment['x0']) - 1.0
            if gap > 1.0:
                model = {
                    'origin_x': float(detection.get('quadratic_global_local', {}).get('origin_x', 0.0)),
                    'coefficients': np.asarray(
                        detection.get('quadratic_global_local', {}).get('coefficients', [0.0, 0.0, 0.0]),
                        dtype=np.float64,
                    ),
                }
                if 'quadratic_global_local' not in detection:
                    merged_model = _fit_merged_detection_model([detection], thickness)
                    if merged_model is None:
                        continue
                    model = merged_model
                evidence = _corridor_support_stats(
                    original_binary,
                    staff_mask,
                    removed_staff_pixels,
                    model,
                    min(selected_x1, segment['x1']) + 1.0,
                    max(selected_x0, segment['x0']) - 1.0,
                    spacing,
                    thickness,
                )
                if not (
                    evidence['coverage'] >= 0.20
                    or (
                        evidence['coverage'] >= 0.08
                        and evidence['staff_fraction'] >= 0.16
                    )
                ):
                    continue

            selected.add(index)
            changed = True

    selected_segments = [segments[index] for index in sorted(selected, key=lambda idx: segments[idx]['x0'])]
    minimum_total_points = max(18, int(round(1.45 * spacing)))
    if sum(len(segment['x']) for segment in selected_segments) < minimum_total_points:
        return None

    return selected_segments


def _copy_observed_curve_pixels(
    detected_mask,
    original_binary,
    segment_points,
    thickness,
):
    points = np.asarray(segment_points, dtype=np.float64)
    if len(points) == 0:
        return
    points = points[np.argsort(points[:, 0])]
    radius = max(1, int(round(0.9 * thickness)))
    height, width = original_binary.shape

    for index, (x_value, y_value) in enumerate(points):
        x = int(round(x_value))
        y = int(round(y_value))
        if x < 0 or x >= width:
            continue
        if index == 0:
            slope = points[min(1, len(points) - 1), 1] - points[0, 1]
        elif index == len(points) - 1:
            slope = points[-1, 1] - points[-2, 1]
        else:
            slope = 0.5 * (points[index + 1, 1] - points[index - 1, 1])

        for dx in (-1, 0, 1):
            sample_x = x + dx
            if sample_x < 0 or sample_x >= width:
                continue
            sample_y = int(round(y + slope * dx))
            y0 = max(0, sample_y - radius)
            y1 = min(height, sample_y + radius + 1)
            detected_mask[y0:y1, sample_x] |= original_binary[y0:y1, sample_x]


def constrain_detections_to_observed_ink(
    detections,
    clean_ink,
    original_binary,
    staff_mask,
    removed_staff_pixels,
    staff_result,
):
    spacing = float(staff_result['metrics']['spacing'])
    thickness = float(staff_result['metrics']['thickness'])
    clean_skeleton = skeletonize(clean_ink > 0).astype(np.uint8)
    degree = convolve(
        clean_skeleton,
        np.ones((3, 3), np.uint8),
        mode='constant',
        cval=0,
    ) - clean_skeleton
    branch_yx = np.argwhere((clean_skeleton > 0) & (degree > 2))
    if branch_yx.size:
        branch_points_xy = np.column_stack([
            branch_yx[:, 1],
            branch_yx[:, 0],
        ]).astype(np.float64)
    else:
        branch_points_xy = np.empty((0, 2), dtype=np.float64)

    accepted = []
    detected_mask = np.zeros_like(original_binary, dtype=np.uint8)

    for detection in detections:
        traced_segments = _trace_curve_columns(
            detection,
            clean_skeleton,
            original_binary,
            staff_mask,
            removed_staff_pixels,
            branch_points_xy,
            spacing,
            thickness,
        )
        if traced_segments is None:
            continue

        curve_segments = []
        source_counts = {'clean': 0, 'staff_recovered': 0}
        all_points = []

        for segment in traced_segments:
            points = np.column_stack([segment['x'], segment['y']])
            points = points[np.argsort(points[:, 0])]
            curve_segments.append(np.rint(points).astype(np.int32).tolist())
            all_points.append(points)
            source_counts['clean'] += int(np.count_nonzero(segment['source'] == 1))
            source_counts['staff_recovered'] += int(np.count_nonzero(segment['source'] == 2))
            _copy_observed_curve_pixels(
                detected_mask,
                original_binary,
                points,
                thickness,
            )

        observed_points = np.vstack(all_points)
        observed_points = observed_points[np.argsort(observed_points[:, 0])]
        item = dict(detection)
        item.pop('staff_crossing_extension', None)
        item.pop('extension_debug', None)
        trajectory_points = np.asarray(detection['curve_points'], dtype=np.float64)
        trajectory_points = trajectory_points[np.argsort(trajectory_points[:, 0])]
        observed_min_x = float(observed_points[:, 0].min())
        observed_max_x = float(observed_points[:, 0].max())
        clip_mask = (
            (trajectory_points[:, 0] >= observed_min_x)
            & (trajectory_points[:, 0] <= observed_max_x)
        )
        clipped_trajectory = trajectory_points[clip_mask]
        if clipped_trajectory.shape[0] < 2:
            clipped_trajectory = observed_points

        item['curve_segments'] = curve_segments
        item['trajectory_curve_points'] = np.rint(clipped_trajectory).astype(np.int32).tolist()
        item['observed_curve_points'] = np.rint(observed_points).astype(np.int32).tolist()
        item['curve_points'] = np.rint(clipped_trajectory).astype(np.int32).tolist()
        item['observed_source_counts'] = source_counts
        item['observed_segment_count'] = len(curve_segments)

        x0 = int(np.floor(clipped_trajectory[:, 0].min()))
        x1 = int(np.ceil(clipped_trajectory[:, 0].max()))
        y0 = int(np.floor(clipped_trajectory[:, 1].min() - 2.0 * thickness))
        y1 = int(np.ceil(clipped_trajectory[:, 1].max() + 2.0 * thickness))
        item['bbox'] = [
            x0,
            y0,
            max(1, x1 - x0 + 1),
            max(1, y1 - y0 + 1),
        ]
        accepted.append(item)

    accepted.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    for index, item in enumerate(accepted):
        item['id'] = index

    return accepted, detected_mask


def _dense_curve_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= 1:
        return np.rint(points).astype(np.int32)
    points = points[np.argsort(points[:, 0])]
    dense = []
    for idx in range(len(points) - 1):
        x0, y0 = points[idx]
        x1, y1 = points[idx + 1]
        steps = max(1, int(round(abs(x1 - x0))))
        xs = np.linspace(x0, x1, steps + 1)
        ys = np.linspace(y0, y1, steps + 1)
        segment = np.column_stack([xs, ys])
        if idx > 0:
            segment = segment[1:]
        dense.append(segment)
    if dense:
        dense = np.vstack(dense)
    else:
        dense = points.copy()
    dense = np.rint(dense).astype(np.int32)
    keep = []
    prev = None
    for pt in dense:
        cur = (int(pt[0]), int(pt[1]))
        if cur != prev:
            keep.append([cur[0], cur[1]])
            prev = cur
    return np.asarray(keep, dtype=np.int32)


def build_reconstructed_curve_mask(shape, detections, staff_result):
    thickness = float(staff_result['metrics']['thickness'])
    draw_thickness = max(1, int(round(max(1.0, thickness))))
    mask = np.zeros(shape, dtype=np.uint8)
    for detection in detections:
        points = detection.get('trajectory_curve_points') or detection.get('curve_points')
        if not points:
            continue
        dense = _dense_curve_points(points)
        if dense.shape[0] < 2:
            continue
        cv2.polylines(mask, [dense.reshape(-1, 1, 2)], False, 255, thickness=draw_thickness, lineType=cv2.LINE_8)
    return mask


def render_reconstructed_curve_overlay(original_bgr, reconstructed_mask, detections):
    overlay = original_bgr.copy()
    active = reconstructed_mask > 0
    overlay[active] = np.array([0, 0, 255], dtype=np.uint8)

    debug = overlay.copy()
    for detection in detections:
        x, y, width, height = detection['bbox']
        cv2.rectangle(debug, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 1)
    return overlay, debug


def restore_detected_curves(clean_ink, reconstructed_mask):
    return np.maximum(clean_ink, reconstructed_mask)


def _handwritten_endpoint_support(
    original_binary,
    global_x,
    global_y,
    opens_downward,
    spacing,
):
    height, width = original_binary.shape
    radius_x = max(4, int(round(0.55 * spacing)))
    depth = max(8, int(round(1.6 * spacing)))

    x0 = max(0, int(round(global_x)) - radius_x)
    x1 = min(width, int(round(global_x)) + radius_x + 1)

    if opens_downward:
        y0 = max(0, int(round(global_y)) + 2)
        y1 = min(height, int(round(global_y)) + depth)
    else:
        y0 = max(0, int(round(global_y)) - depth)
        y1 = min(height, int(round(global_y)) - 2)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    region = original_binary[y0:y1, x0:x1] > 0
    return float(np.mean(region))


def _handwritten_curve_duplicate(first, second, spacing):
    first_points = np.asarray(first['curve_points'], dtype=np.float64)
    second_points = np.asarray(second['curve_points'], dtype=np.float64)

    first_x0 = float(first_points[:, 0].min())
    first_x1 = float(first_points[:, 0].max())
    second_x0 = float(second_points[:, 0].min())
    second_x1 = float(second_points[:, 0].max())

    overlap = min(first_x1, second_x1) - max(first_x0, second_x0)
    if overlap <= 0:
        return False

    first_width = max(1.0, first_x1 - first_x0)
    second_width = max(1.0, second_x1 - second_x0)
    overlap_ratio = overlap / min(first_width, second_width)

    if overlap_ratio < 0.7:
        return False

    sample_x = np.linspace(
        max(first_x0, second_x0),
        min(first_x1, second_x1),
        32,
    )
    first_y = np.interp(
        sample_x,
        first_points[:, 0],
        first_points[:, 1],
    )
    second_y = np.interp(
        sample_x,
        second_points[:, 0],
        second_points[:, 1],
    )

    return float(np.median(np.abs(first_y - second_y))) < 0.45 * spacing


def detect_handwritten_curves(
    clean_ink,
    original_binary,
    staff_result,
):
    spacing = float(staff_result['metrics']['spacing'])
    thickness = float(staff_result['metrics']['thickness'])
    binary = (clean_ink > 0).astype(np.uint8)

    vertical_length = max(
        9,
        int(round(0.8 * spacing)),
    )
    vertical_strokes = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, vertical_length),
        ),
    )

    work = binary.copy()
    work[vertical_strokes > 0] = 0
    work = cv2.morphologyEx(
        work,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )

    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            work,
            connectivity=8,
        )
    )

    minimum_width = max(
        18,
        int(round(1.25 * spacing)),
    )
    maximum_height = max(
        90,
        int(round(4.5 * spacing)),
    )
    band_margin = 2.2 * spacing
    clip_px = max(
        3.0,
        1.8 * thickness,
    )

    candidates = []

    for component_id in range(1, component_count):
        x, y, width, height, area = (
            int(value)
            for value in stats[component_id]
        )

        if width < minimum_width:
            continue
        if height < 2 or height > maximum_height:
            continue
        if width / max(1, height) < 1.3:
            continue
        if area / max(1, width * height) > 0.48:
            continue

        center_y = y + 0.5 * height
        inside_staff_band = any(
            stave['yTop'] - band_margin
            <= center_y
            <= stave['yBottom'] + band_margin
            for stave in staff_result['staves']
        )
        if not inside_staff_band:
            continue

        component = (
            labels[
                y:y + height,
                x:x + width,
            ]
            == component_id
        )

        component_candidates = []

        for mode in ('top', 'bottom', 'median'):
            sequence = component_boundary_sequence(
                component,
                mode,
            )
            fit = robust_quadratic_fit(
                sequence,
                clip_px,
            )
            if fit is None:
                continue

            local_width = (
                int(fit['x1'])
                - int(fit['x0'])
                + 1
            )
            coefficients = fit['coefficients']
            slope_start = float(
                2.0
                * coefficients[0]
                * fit['x0']
                + coefficients[1]
            )
            slope_end = float(
                2.0
                * coefficients[0]
                * fit['x1']
                + coefficients[1]
            )
            sag_ratio = float(
                fit['sag']
                / max(1, local_width)
            )

            if local_width < minimum_width:
                continue
            if fit['coverage'] < 0.42:
                continue
            if fit['rmse'] > max(
                4.5,
                2.4 * thickness,
            ):
                continue
            if fit['sag'] < max(
                3.0,
                0.15 * spacing,
            ):
                continue
            if sag_ratio < 0.022:
                continue
            if fit['sag'] > max(
                80.0,
                4.5 * spacing,
            ):
                continue
            if max(
                abs(slope_start),
                abs(slope_end),
            ) > 1.8:
                continue
            if (
                slope_start * slope_end > 0
                and min(
                    abs(slope_start),
                    abs(slope_end),
                ) > 0.18
            ):
                continue

            score = float(
                local_width * fit['coverage']
                + 4.0 * fit['sag']
                - 8.0 * fit['rmse']
            )
            if score < 65.0:
                continue

            left_x = float(x + fit['x0'])
            right_x = float(x + fit['x1'])
            left_y = float(
                y
                + np.polyval(
                    coefficients,
                    fit['x0'],
                )
            )
            right_y = float(
                y
                + np.polyval(
                    coefficients,
                    fit['x1'],
                )
            )

            opens_downward = bool(
                coefficients[0] < 0
            )
            left_support = _handwritten_endpoint_support(
                original_binary,
                left_x,
                left_y,
                opens_downward,
                spacing,
            )
            right_support = _handwritten_endpoint_support(
                original_binary,
                right_x,
                right_y,
                opens_downward,
                spacing,
            )
            minimum_support = min(
                left_support,
                right_support,
            )
            support_threshold = (
                0.025
                if score >= 150.0
                else 0.04
            )
            if minimum_support < support_threshold:
                continue

            sample_count = min(
                240,
                max(40, local_width),
            )
            sample_x = np.linspace(
                fit['x0'],
                fit['x1'],
                sample_count,
            )
            sample_y = np.polyval(
                coefficients,
                sample_x,
            )
            curve_points = np.column_stack([
                x + sample_x,
                y + sample_y,
            ])

            candidate = {
                'id': -1,
                'type': 'slur_or_tie',
                'source': 'handwritten_component',
                'component_id': component_id,
                'boundary_mode': mode,
                'bbox': [
                    int(np.floor(curve_points[:, 0].min())),
                    int(np.floor(
                        curve_points[:, 1].min()
                        - 2.0 * thickness
                    )),
                    int(np.ceil(
                        curve_points[:, 0].max()
                        - curve_points[:, 0].min()
                        + 1
                    )),
                    int(np.ceil(
                        curve_points[:, 1].max()
                        - curve_points[:, 1].min()
                        + 4.0 * thickness
                        + 1
                    )),
                ],
                'coverage': float(fit['coverage']),
                'rmse': float(fit['rmse']),
                'sag': float(fit['sag']),
                'sag_ratio': sag_ratio,
                'slope_start': slope_start,
                'slope_end': slope_end,
                'endpoint_support': {
                    'left': left_support,
                    'right': right_support,
                },
                'score': score,
                'curve_points': (
                    np.rint(curve_points)
                    .astype(np.int32)
                    .tolist()
                ),
            }
            component_candidates.append(candidate)

        if component_candidates:
            candidates.append(
                max(
                    component_candidates,
                    key=lambda item: item['score'],
                )
            )

    candidates.sort(
        key=lambda item: item['score'],
        reverse=True,
    )

    kept = []

    for candidate in candidates:
        if any(
            _handwritten_curve_duplicate(
                candidate,
                other,
                spacing,
            )
            for other in kept
        ):
            continue
        kept.append(candidate)

    kept.sort(
        key=lambda item: (
            item['bbox'][1],
            item['bbox'][0],
        )
    )

    for index, candidate in enumerate(kept):
        candidate['id'] = index

    return kept

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--page', type=int, default=1)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    bgr, natural_width, natural_height = render_pdf(
        args.pdf,
        args.page - 1,
        args.dpi,
    )

    staff_result, gray, bin_img, line_ink = detect_stafflines(bgr)
    staff_result['naturalW'] = natural_width
    staff_result['naturalH'] = natural_height

    clean_ink, staff_mask, removed_pixels = make_cleanplate(
        gray,
        bin_img,
        staff_result,
    )

    base_overlay, base_debug_overlay, base_detections = detect_slurs_and_ties(
        clean_ink,
        bgr,
        staff_result,
    )
    original_binary=(bin_img*255).astype(np.uint8)
    skeleton_detections = detect_skeleton_curve_chains(
        clean_ink,
        original_binary,
        staff_result,
    )
    detections = combine_curve_detections(
        base_detections,
        skeleton_detections,
        original_binary,
        staff_result,
    )
    detections = merge_curve_fragments(
        detections,
        original_binary,
        staff_mask,
        removed_pixels,
        staff_result,
    )
    detections, detected_curve_mask = constrain_detections_to_observed_ink(
        detections,
        clean_ink,
        original_binary,
        staff_mask,
        removed_pixels,
        staff_result,
    )

    handwritten_detections = detect_handwritten_curves(
        clean_ink,
        original_binary,
        staff_result,
    )
    detection_mode = 'printed'

    if len(handwritten_detections) >= max(
        12,
        2 * len(detections),
    ):
        detections = handwritten_detections
        detection_mode = 'handwritten'
        detected_curve_mask = np.zeros_like(
            original_binary,
            dtype=np.uint8,
        )

    reconstructed_curve_mask = build_reconstructed_curve_mask(
        original_binary.shape,
        detections,
        staff_result,
    )
    overlay, debug_overlay = render_reconstructed_curve_overlay(
        bgr,
        reconstructed_curve_mask,
        detections,
    )
    restored_clean_ink = restore_detected_curves(
        clean_ink,
        reconstructed_curve_mask,
    )

    cv2.imwrite(str(args.outdir / 'render.png'), bgr)
    cv2.imwrite(str(args.outdir / 'staff_overlay.png'), staff_overlay(bgr, staff_result))
    cv2.imwrite(str(args.outdir / 'cleanplate.png'), 255 - clean_ink)
    cv2.imwrite(str(args.outdir / 'cleanplate_slurs_restored.png'), 255 - restored_clean_ink)
    cv2.imwrite(str(args.outdir / 'staff_mask.png'), staff_mask)
    cv2.imwrite(str(args.outdir / 'removed_staff_pixels.png'), removed_pixels)
    cv2.imwrite(str(args.outdir / 'detected_curve_mask.png'), detected_curve_mask * 255)
    cv2.imwrite(str(args.outdir / 'reconstructed_curve_mask.png'), reconstructed_curve_mask)
    cv2.imwrite(str(args.outdir / 'slur_tie_overlay.png'), overlay)
    cv2.imwrite(str(args.outdir / 'slur_tie_overlay_debug.png'), debug_overlay)

    payload = {
        'source_pdf': str(args.pdf),
        'page': args.page,
        'dpi': args.dpi,
        'coordinate_space': 'downscaled_detection_space',
        'natural_size': {
            'width': natural_width,
            'height': natural_height,
        },
        'detection_size': {
            'width': int(bgr.shape[1]),
            'height': int(bgr.shape[0]),
        },
        'staff_result': staff_result,
        'slur_tie_count': len(detections),
        'detection_mode': detection_mode,
        'handwritten_candidate_count': len(handwritten_detections),
        'component_boundary_count': len(base_detections),
        'skeleton_chain_count': len(skeleton_detections),
        'slur_ties': detections,
    }

    (args.outdir / 'detections.json').write_text(
        json.dumps(payload, indent=2),
        encoding='utf-8',
    )

    print(json.dumps({
        'staves': len(staff_result['staves']),
        'slur_ties': len(detections),
        'staff_spacing': staff_result['metrics']['spacing'],
        'staff_thickness': staff_result['metrics']['thickness'],
        'detection_mode': detection_mode,
        'handwritten_candidates': len(handwritten_detections),
    }))


if __name__ == '__main__':
    main()
