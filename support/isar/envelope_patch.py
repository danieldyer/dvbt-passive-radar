"""Physical motion likelihood with shared phases and positive RCS envelopes."""
import numpy as np
from audit_ship_scattering_envelopes import linear_basis,statistics
from positive_envelope_scene import solve
from rigid_patch import profile_static
from motion import excess_path


class EnvelopePatch:
    def __init__(self,bands,interval,spacing,*,refine_positions=False,position_phase_gauge=True):
        self.bands=bands;self.points=bands[0].points;self.npoints=len(self.points);self.reference=min(3,len(bands)-1)
        self.free=[c for c in range(len(bands)) if c!=self.reference];self.phase_end=5+len(self.free)+self.npoints
        self.refine_positions=refine_positions;self.size=self.phase_end+(2*(self.npoints-1) if refine_positions else 0)
        self.bases=[linear_basis(b.field.time,interval,spacing)[0] for b in bands];self.knots=linear_basis(bands[0].field.time,interval,spacing)[1]
        self.groups=np.repeat(np.arange(self.npoints),len(self.knots));self.energy=sum(b.energy for b in bands)
        self.position_phase_gauge=np.zeros((self.npoints,2))
        if refine_positions and position_phase_gauge:
            ref=bands[self.reference];kin=ref.kinematics;at=int(np.argmin(abs(kin.time-kin.epoch)))
            position=kin.position[at]+self.points@kin.frame[at].T
            _,g=excess_path(position,ref.tx,ref.rx)
            self.position_phase_gauge=ref.field.k*(g@kin.frame[at,:,:2])

    def evaluate(self,p,*,derivatives=True):
        motion=p[:5];beta=np.zeros(len(self.bands));beta[self.free]=p[5:5+len(self.free)];phi=p[5+len(self.free):self.phase_end]
        points=self.points.copy()
        if self.refine_positions:
            offset=np.asarray(p[self.phase_end:]).reshape(-1,2);points[1:,:2]+=offset
            phi=phi.copy();phi[1:]+=np.sum(offset*self.position_phase_gauge[1:],axis=1)
        cost=0.;gradient=np.zeros(self.size);solutions=[];designs=[]
        for c,(band,basis) in enumerate(zip(self.bands,self.bases)):
            h,dh=band.design(motion,points=points,derivatives=derivatives,point_derivatives=self.refine_positions);f=band.field;mask=band.training
            gram,rhs,energy=statistics(h,f.y,f.root_information,mask,basis)
            solution=solve(gram,rhs,energy,self.groups,phi+beta[c]);solutions.append(solution);designs.append(h)
            cost+=solution['cost'];gradient[5+len(self.free):self.phase_end]+=solution['phase_gradient']
            if c in self.free:gradient[5+self.free.index(c)]+=sum(solution['phase_gradient'])
            if derivatives:
                coefficients=basis@solution['coefficient'].reshape(self.npoints,len(self.knots)).T
                predicted=np.einsum('trp,tp->tr',h,coefficients)
                residual=(profile_static(f.y-predicted,f.root_information,mask)*mask[:,None])
                change=np.einsum('trpj,tp->trj',dh[:,:,:,:5],coefficients)
                gradient[:5]-=2*np.einsum('tr,trj->j',residual.conj(),change).real
                if self.refine_positions:
                    point_gradient=-2*np.einsum('tr,trpj,tp->pj',residual.conj(),dh[:,:,:,5:],coefficients).real
                    gradient[self.phase_end:]+=point_gradient[1:].ravel()
        if self.refine_positions:
            gradient[self.phase_end:]+=(gradient[5+len(self.free)+1:self.phase_end,None]*self.position_phase_gauge[1:]).ravel()
        return dict(value=float(cost/self.energy),gradient=gradient/self.energy,solutions=solutions,designs=designs,points=points,point_phase_radians=phi)

    def validation(self,p,evaluation=None):
        fitted=self.evaluate(p,derivatives=False) if evaluation is None else evaluation;reports=[]
        for band,basis,h,solution in zip(self.bands,self.bases,fitted['designs'],fitted['solutions']):
            f=band.field;mask=~band.training;coefficients=basis@solution['coefficient'].reshape(self.npoints,len(self.knots)).T
            prediction=np.einsum('trp,tp->tr',h,coefficients);data=profile_static(f.y,f.root_information,mask)[mask]
            predicted=profile_static(prediction,f.root_information,mask)[mask];residual=data-predicted
            reports.append(dict(unused_rss=float(np.vdot(residual,residual).real),unused_data_energy=float(np.vdot(data,data).real),
                unused_prediction_energy=float(np.vdot(predicted,predicted).real),unused_cross_energy=float(np.vdot(predicted,data).real)))
        return reports
