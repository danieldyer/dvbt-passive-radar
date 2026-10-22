"""Runtime Vulkan image/gradient checks against the independent CPU operator."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from block_pulse import BlockPulse
from dense_patch_image import DenseImages,entropy,trial_geometry
from rigid_patch import PatchBand
from test_rigid_patch import scenario
from waveform_kernels import KernelField


class DenseFocusValidation(unittest.TestCase):
    def test_recorded_operator_polynomials_image_and_all_five_pose_derivatives(self):
        bands=[];points=np.array([[0.,0.,20.],[80.,25.,20.],[-70.,-30.,20.],[10.,15.,20.]])
        for rf,g in [(177.5e6,.6+.2j),(226.5e6,-.4+.8j)]:
            kin,b,p=scenario(rf,g)
            kernels=BlockPulse(np.repeat(b.field.pulse.samples[None,:],len(b.field.time),axis=0))
            field=KernelField(b.field,kernels,np.arange(len(b.field.time)))
            bands.append(PatchBand(field,kin,b.points,b.training,b.tx,b.rx))
        with tempfile.TemporaryDirectory(prefix='dense-focus-validation-',dir='build') as directory:
            model=DenseImages(bands,points,Path(directory))
            try:
                actual=model.evaluate(p);reference=[]
                for band in bands:
                    h,dh=band.design(p,derivatives=True,points=points)
                    y=band.project(band.field.y)*band.training[:,None]
                    z=np.einsum('trp,tr->p',h.conj(),y)
                    j=np.einsum('trpj,tr->jp',dh.conj(),y)
                    reference.append(np.r_[z[None],j])
                reference=np.asarray(reference)
                image_error=np.linalg.norm(actual[:,0]-reference[:,0])/np.linalg.norm(reference[:,0])
                derivative_errors=[np.linalg.norm(actual[:,j+1]-reference[:,j+1])/np.linalg.norm(reference[:,j+1]) for j in range(5)]
                phase=np.degrees(np.angle(actual[:,0]/reference[:,0]))
                print('Dense Vulkan image relative complex error:',image_error,flush=True)
                print('Dense Vulkan five derivative relative errors:',derivative_errors,flush=True)
                print('Dense Vulkan maximum image phase error (degrees):',abs(phase).max(),flush=True)
                self.assertLess(image_error,5e-4);self.assertLess(max(derivative_errors),8e-4);self.assertLess(abs(phase).max(),.05)
                histories=model.histories()
                np.testing.assert_allclose(histories.sum(axis=1),actual[:,0],rtol=5e-6,atol=.05)
                for c,band in enumerate(bands):
                    h,_=band.design(p,points=points)
                    y=band.project(band.field.y)*band.training[:,None]
                    expected_rows=np.einsum('trp,tr->tp',h.conj(),y)
                    np.testing.assert_allclose(histories[c,:len(expected_rows)],expected_rows,rtol=5e-4,atol=.01)
                    np.testing.assert_array_equal(histories[c,len(expected_rows):],0.)
                # Returning to the usual image/adjoint protocol after a
                # history request must not retain an incompatible buffer.
                np.testing.assert_allclose(model.evaluate(p),actual,rtol=1e-5,atol=.1)
                rng=np.random.default_rng(781)
                weight=rng.normal(size=actual[:,0].shape)+1j*rng.normal(size=actual[:,0].shape)
                adjoint=model.adjoint(weight).sum(axis=1)
                expected=2*np.real(np.einsum('cp,cjp->cj',weight.conj(),actual[:,1:]))
                error=np.linalg.norm(adjoint-expected)/np.linalg.norm(expected)
                print('Dense Vulkan adjoint dot product relative error:',error,flush=True)
                self.assertLess(error,1e-4)
                # Band order, padding and zero-information epochs all enter
                # the CPU comparison above; a second trial catches stale data.
                changed=model.evaluate(p+np.array([.01,0.,0.,.02,0.]))
                self.assertGreater(np.linalg.norm(changed[:,0]-actual[:,0])/np.linalg.norm(actual[:,0]),.01)
            finally:model.close()

    def test_row_local_geometry_and_continuous_motion_adjoint(self):
        from ship_window import local_geometry
        from spline_motion import basis
        kin,b,p=scenario();kernels=BlockPulse(np.repeat(b.field.pulse.samples[None,:],len(b.field.time),axis=0))
        field=KernelField(b.field,kernels,np.arange(len(b.field.time)))
        band=PatchBand(field,kin,b.points,b.training,b.tx,b.rx);t=field.time;dt=t-kin.epoch
        path=p[0]*dt+p[1]*dt**2/2+p[2]*dt**3/6;rate=p[0]+p[1]*dt+p[2]*dt**2/2
        angle=np.radians(p[3]*dt+p[4]*dt**2/2);angular_rate=np.radians(p[3]+p[4]*dt)
        points=np.array([[0.,0.,20.],[80.,25.,20.],[-70.,-30.,20.]])
        with tempfile.TemporaryDirectory(prefix='local-focus-validation-',dir='build') as directory:
            model=DenseImages([band],points,Path(directory))
            try:
                expected=model.evaluate(p)
                geo=local_geometry(band,path,rate,angle,angular_rate,model.selected[0],model.stride)
                actual=model.evaluate_geometry(geo[None])
                np.testing.assert_allclose(actual[:,0],expected[:,0],rtol=1e-5,atol=1e-2)
                # Chain the local path/rate/yaw/yaw-rate derivative through
                # the same polynomial used by the independently tested pose.
                rng=np.random.default_rng(392);weight=rng.normal(size=(1,len(points)))+1j*rng.normal(size=(1,len(points)))
                local=model.adjoint(weight)[0,:len(t)]
                h=np.c_[dt,dt**2/2,dt**3/6];hd=np.c_[np.ones(len(dt)),dt,dt**2/2]
                angular=np.radians(np.c_[dt,dt**2/2]);ad=np.radians(np.c_[np.ones(len(dt)),dt])
                chain=np.r_[h.T@local[:,0]+hd.T@local[:,1],angular.T@local[:,2]+ad.T@local[:,3]]
                reference=2*np.real(np.einsum('cp,cjp->j',weight.conj(),expected[:,1:]))
                error=np.linalg.norm(chain-reference)/np.linalg.norm(reference)
                print('Row-local motion chain-rule relative error:',error,flush=True)
                self.assertLess(error,2e-5)
            finally:model.close()

    def test_eight_variable_packet_retains_five_variable_results(self):
        kin,b,p=scenario();kernels=BlockPulse(np.repeat(b.field.pulse.samples[None,:],len(b.field.time),axis=0))
        band=PatchBand(KernelField(b.field,kernels,np.arange(len(b.field.time))),kin,b.points,b.training,b.tx,b.rx)
        points=np.array([[0.,0.,20.],[80.,25.,20.],[-70.,-30.,20.]])
        with tempfile.TemporaryDirectory(prefix='eight-focus-validation-',dir='build') as directory:
            five=DenseImages([band],points,Path(directory))
            try:
                expected=five.evaluate(p);geometry=trial_geometry(band,p,five.selected[0],five.stride)
            finally:five.close()
            eight=DenseImages.reopen(Path(directory)/'model.bin',derivatives=8)
            packet=np.zeros((1,eight.stride,74,4),'<f4');packet[0,:,:50]=geometry
            try:
                actual=eight.evaluate_geometry(packet)
                np.testing.assert_allclose(actual[:,:6],expected,rtol=5e-5,atol=.1)
                np.testing.assert_array_equal(actual[:,6:],0.)
                image_only=eight.evaluate_geometry(packet,derivatives=False)
                np.testing.assert_allclose(image_only[:,0],actual[:,0],rtol=5e-6,atol=.1)
                eight.set_pixels(points[:2])
                subset=eight.evaluate_cached()
                np.testing.assert_allclose(subset,actual[:,:,:2],rtol=5e-6,atol=.1)
                eight.set_pixels(points)
                actual=eight.evaluate_cached()
                import struct
                cached=Path(directory)/'model.bin';body=bytearray(cached.read_bytes())
                count=len(eight.bands)*eight.stride*eight.cells*(eight.degree+1)
                coefficients=np.frombuffer(body,dtype='<c8',offset=32,count=count);coefficients*=2.
                replacement=Path(directory)/'scaled-model.bin';replacement.write_bytes(body)
                eight.replace_archive(replacement);changed=eight.evaluate_geometry(packet)
                np.testing.assert_allclose(changed,2*actual,rtol=5e-6,atol=.1)
                eight.replace_archive(cached);actual=eight.evaluate_geometry(packet)
                rng=np.random.default_rng(88);weight=rng.normal(size=(1,3))+1j*rng.normal(size=(1,3))
                adjoint=eight.adjoint(weight).sum(axis=1)
                reference=2*np.real(np.einsum('cp,cjp->cj',weight.conj(),actual[:,1:]))
                np.testing.assert_allclose(adjoint,reference,rtol=3e-5,atol=.1)
            finally:eight.close()

    def test_transmitter_displacement_changes_range_carrier_and_doppler(self):
        from ship_window import local_geometry
        from motion import excess_path
        kin,b,p=scenario();kernels=BlockPulse(np.repeat(b.field.pulse.samples[None,:],len(b.field.time),axis=0))
        field=KernelField(b.field,kernels,np.arange(len(b.field.time)))
        band=PatchBand(field,kin,b.points,b.training,b.tx,b.rx);t=field.time
        path=p[0]*t+p[1]*t**2/2+p[2]*t**3/6;rate=p[0]+p[1]*t+p[2]*t**2/2
        angle=np.radians(p[3]*t+p[4]*t**2/2);ar=np.radians(p[3]+p[4]*t)
        point=np.array([-5000.,-6500.,10.]);original_tx=b.tx.copy();axis=np.array([.6,.8,0.])
        def geometry(offset,model):
            band.tx=original_tx+offset*axis
            result=local_geometry(band,path,rate,angle,ar,model.selected[0],model.stride)
            reference=(excess_path(point[None],band.tx,b.rx)[0]-excess_path(point[None],original_tx,b.rx)[0])[0]
            z=(result[:len(t),9,1]+1j*result[:len(t),9,2])*np.exp(-1j*field.k*reference)
            result[:len(t),9,1]=z.real;result[:len(t),9,2]=z.imag
            u=(point-band.tx)/np.linalg.norm(point-band.tx);baseline=(band.tx-b.rx)/np.linalg.norm(band.tx-b.rx)
            result[:len(t),26,:3]=axis;result[:len(t),26,3]=1.
            result[:len(t),27,:3]=baseline;result[:len(t),27,3]=(-u-baseline)@axis
            result[:len(t),28:34]=0.
            return result[None]
        with tempfile.TemporaryDirectory(prefix='antenna-focus-validation-',dir='build') as directory:
            model=DenseImages([band],np.array([[0.,0.,20.],[80.,25.,20.],[-70.,-30.,20.]]),Path(directory))
            try:
                expected=model.evaluate_geometry(geometry(0.,model))[:,3]
                h=1.;plus=model.evaluate_geometry(geometry(h,model))[:,0];minus=model.evaluate_geometry(geometry(-h,model))[:,0]
                finite=(plus-minus)/(2*h);error=np.linalg.norm(expected-finite)/np.linalg.norm(expected)
                print('Transmitter finite-difference relative gradient error:',error,flush=True)
                self.assertLess(error,.004)
                band.tx=original_tx
                baseline=(original_tx-b.rx)/np.linalg.norm(original_tx-b.rx)
                u=(point-original_tx)/np.linalg.norm(point-original_tx);reference_derivative=(-u-baseline)@axis
                y=band.project(field.y)*band.training[:,None];cpu=[]
                for scatterer in model.pixels:
                    pos,vel=kin.point_state(p,scatterer)
                    raw,derivative=field.raw_trajectory(pos,vel,transmitter=original_tx,receiver=b.rx,antenna_derivatives=True)
                    value=field.apply(raw);change=field.apply(np.einsum('trj,j->tr',derivative[:,:,:3],axis))
                    cpu.append(np.vdot(change,y)-1j*field.k*reference_derivative*np.vdot(value,y))
                reference=np.asarray(cpu)[None]
                exact_error=np.linalg.norm(expected-reference)/np.linalg.norm(reference)
                print('Transmitter independent CPU Jacobian relative error:',exact_error,flush=True)
                self.assertLess(exact_error,5e-4)
            finally:model.close()

    def test_image_entropy_derivative_is_not_a_pixelwise_phase_adjustment(self):
        rng=np.random.default_rng(37);image=rng.normal(size=71)+1j*rng.normal(size=71)
        derivative=rng.normal(size=(71,5))+1j*rng.normal(size=(71,5));value,gradient=entropy(image,derivative)
        for j in range(5):
            finite=(entropy(image+1e-5*derivative[:,j])-entropy(image-1e-5*derivative[:,j]))/2e-5
            self.assertAlmostEqual(finite,gradient[j],places=8)


if __name__=='__main__':unittest.main()
