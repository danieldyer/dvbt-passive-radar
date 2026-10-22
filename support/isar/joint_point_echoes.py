"""One geometric point plus separately acquired neighbouring path histories.

Every echo traverses the saved waveform, range migration, RF phase, C0/C1
moments and original-row averaging. A neighbour has a quadratic excess path
relative to the declared AIS reference; its constant band coefficient is
profiled jointly with the point. No phase screen is applied to the data.
"""
import numpy as np
from scipy.optimize import least_squares

from field import average, FS
from motion import C
from recovery_experiment import Experiment, SCALE

PATH_SCALE = np.array([100., .5, .02])


class JointPointEchoes(Experiment):
    def __init__(self, base, count=1):
        self.__dict__.update(base.__dict__)
        if count < 1:
            raise ValueError('At least one additional echo is required')
        self.count = count
        self.parameter_count = 6+3*count*len(self.fields)

    def indices(self, channel, echo):
        start = 6+3*(channel*self.count+echo)
        return slice(start,start+3)

    def path_template(self, parameters, channel, *, derivatives=False):
        f = self.fields[channel]; d = f.data
        t = d['row_time']-self.midpoint
        basis = np.column_stack([np.ones(len(t)),t,.5*t*t])*PATH_SCALE
        rate_basis = np.column_stack([np.zeros(len(t)),np.ones(len(t)),t])*PATH_SCALE
        delta = basis@parameters
        omega = d['centre_omega']-f.k*.001024*(rate_basis@parameters)
        if derivatives:
            h, dl, dw, _ = f.path_response(delta,omega,derivatives=True)
            dh = dl[:,:,None]*basis[:,None,:]-dw[:,:,None]*f.k*.001024*rate_basis[:,None,:]
            dh = average(dh,d['row_information'],f.metadata['block_rows'])[0]
            jac = np.stack([f.apply(dh[:,:,j]).ravel() for j in range(3)],axis=-1)
        else:
            h = f.path_response(delta,omega)
        h = f.apply(average(h,d['row_information'],f.metadata['block_rows'])[0]).ravel()
        return (h,jac) if derivatives else h

    def templates(self, parameters, c, *, derivatives=False, offsets=None):
        if offsets is not None:
            raise ValueError('Additional point support must be declared explicitly')
        point = super().templates(parameters[:6],c,derivatives=derivatives)
        if derivatives:
            h,j = point
            dh = np.zeros((len(h),1+self.count,self.parameter_count),complex)
            dh[:,0,:6] = j[:,0]
        else:
            h = point
        values = [h[:,0]]
        for e in range(self.count):
            indices = self.indices(c,e)
            value = self.path_template(parameters[indices],c,derivatives=derivatives)
            if derivatives:
                v,j = value; dh[:,1+e,indices] = j
            else:
                v = value
            values.append(v)
        a = np.column_stack(values)
        return (a,dh) if derivatives else a

    def acquire_neighbour(self, y, p, channel):
        """Training-only chirp search after projecting the fitted point/static span."""
        f = self.fields[channel]; keep = self.training[channel]
        use = np.repeat(keep,len(f.offsets))
        point = Experiment.templates(self,p[:6],channel)[use]
        q,_ = np.linalg.qr(np.column_stack([self.q[channel],point]),mode='reduced')
        clean = y[use]-q@(q.conj().T@y[use])
        delays = np.arange(-8.,8.01,.5)*C/FS
        scores = []
        for delay in delays:
            h = self.path_template(np.array([delay/100.,0.,0.]),channel)[use].reshape(-1,len(f.offsets))
            scores.append(np.sum(abs(np.sum(h.conj()*clean.reshape(h.shape),axis=1))**2/
                np.maximum(np.sum(abs(h)**2,axis=1),1e-100)))
        chosen = []
        for i in np.argsort(scores)[::-1]:
            if all(abs(delays[i]-delays[j])>C/FS for j in chosen): chosen.append(i)
            if len(chosen)==3: break
        rates, acc = np.meshgrid(np.arange(-1.,1.001,.025),np.arange(-.03,.0301,.002),indexing='ij')
        rates,acc = rates.ravel(),acc.ravel()
        dt = np.repeat(f.time[keep]-self.midpoint,len(f.offsets))
        best = None
        for di in chosen:
            base = self.path_template(np.array([delays[di]/100.,0.,0.]),channel)[use]
            for first in range(0,len(rates),128):
                rr,aa = rates[first:first+128],acc[first:first+128]
                h = base[:,None]*np.exp(-1j*f.k*(dt[:,None]*rr+.5*dt[:,None]**2*aa))
                norm = np.sum(abs(h)**2,axis=0)-np.sum(abs(q.conj().T@h)**2,axis=0)
                score = abs(h.conj().T@clean)**2/np.maximum(norm,1e-100)
                j = int(np.argmax(score))
                if best is None or score[j]>best['score']:
                    best = dict(parameters=np.array([delays[di],rr[j],aa[j]])/PATH_SCALE,
                        score=float(score[j]))
        return best

    def fit(self,y,seed,max_evaluations=100,free=None,*,position_halfwidth_m=1000.):
        free = np.arange(self.parameter_count) if free is None else np.asarray(free)
        limit = np.r_[position_halfwidth_m/SCALE[:2],2.,2.,2.,2.,np.tile([4.,4.,5.],self.count*len(self.fields))]
        last = [None,None]
        def evaluate(q):
            p = np.asarray(seed).copy(); p[free] = q
            if last[0] is None or not np.array_equal(last[0],q): last[:] = [q.copy(),self.profile(p,y)]
            return last[1]
        fit = least_squares(lambda q:evaluate(q)[0],np.asarray(seed)[free],jac=lambda q:evaluate(q)[1][:,free],
            bounds=(-limit[free],limit[free]),x_scale='jac',ftol=1e-11,xtol=1e-11,gtol=1e-8,max_nfev=max_evaluations)
        p = np.asarray(seed).copy(); p[free] = fit.x
        r,j,alpha,prediction = self.profile(p,y)
        return dict(parameters=p,training_rss=r@r,test_rss=[np.sum(abs((v-w).reshape(f.y.shape)[~train])**2)
                for v,w,f,train in zip(y,prediction,self.fields,self.training)],
            success=fit.success,message=fit.message,nfev=fit.nfev,gradient=fit.optimality,
            singular_values=np.linalg.svd(j[:,free],compute_uv=False),
            coefficients_real_imag=[np.c_[a.real,a.imag] for a in alpha]),prediction
