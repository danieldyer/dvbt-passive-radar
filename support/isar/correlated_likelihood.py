"""Finite-aperture feasible GLS for the existing physical point experiment.

Only the likelihood changes: a frozen positive covariance acts equally on the
observations, templates, nuisance columns and analytic derivatives. Fourier
coordinates are unitary. This support estimator assumes a block-circulant
background and does not claim that an arbitrary recorded scene is stationary.
"""
import numpy as np

from recovery_experiment import Experiment


def sample_covariance(samples, *, shrink=True):
    """Complex sample covariance with estimated quadratic-risk shrinkage.

    Samples are rows, with E[x x^H] as the covariance convention. The target
    is trace(S)/p I. The estimated variance of S uses independent zero-mean
    outer-product samples; its validity needs checking on recorded residuals.
    A unit eigenvalue floor retains the declared conditional thermal floor.
    """
    x = np.asarray(samples)
    n, p = x.shape
    if n < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite covariance samples')
    s = x.T@x.conj()/n
    trace = np.trace(s).real
    target = np.eye(p)*trace/p
    variance = max(0., (np.mean(np.sum(abs(x)**2, axis=1)**2)-np.sum(abs(s)**2))/(n-1))
    distance = np.sum(abs(s-target)**2)
    amount = min(1., variance/distance) if distance > 0 else 1.
    if shrink:
        s = (1-amount)*s+amount*target
    eigen, vectors = np.linalg.eigh((s+s.conj().T)/2)
    return (vectors*np.maximum(eigen, 1.))@vectors.conj().T, amount


class FourierMetric:
    """A fixed, invertible inverse covariance square root for one aperture."""
    def __init__(self, labels, covariance):
        self.labels = np.asarray(labels)
        self.covariance = np.asarray(covariance)
        eigen, vectors = np.linalg.eigh(self.covariance)
        if np.any(eigen <= 0):
            raise ValueError('The likelihood covariance must be positive definite')
        self.inverse_root = (vectors/eigen[:, None, :]**.5)@vectors.conj().transpose(0, 2, 1)

    def apply(self, values):
        values = np.asarray(values)
        n, r = len(self.labels), self.covariance.shape[-1]
        original_shape = values.shape
        value = np.fft.fft(values.reshape(n, r, -1), axis=0, norm='ortho')
        result = np.einsum('nrs,nsk->nrk', self.inverse_root[self.labels], value)
        return result.reshape(original_shape)

    @classmethod
    def estimate(cls, residual, dt_s, width_hz=2., *, scalar=False):
        n, r = residual.shape
        frequency = np.fft.fftfreq(n, dt_s)
        bins = np.floor(frequency/width_hz).astype(int)
        _, labels = np.unique(bins, return_inverse=True)
        spectrum = np.fft.fft(residual, axis=0, norm='ortho')
        covariance, shrinkage, counts = [], [], []
        for b in range(labels.max()+1):
            x = spectrum[labels == b]
            if len(x) < 2:
                raise ValueError('A covariance band has too few frequency samples')
            c, amount = sample_covariance(x)
            if scalar:
                c = np.eye(r)*np.trace(c).real/r
            covariance.append(c); shrinkage.append(amount); counts.append(len(x))
        result = cls(labels, covariance)
        result.description = dict(width_hz=width_hz, scalar=scalar, counts=counts,
            shrinkage=shrinkage, eigenvalues=np.linalg.eigvalsh(covariance).tolist(),
            assumption='Independent zero-mean Fourier vectors with a constant range covariance in each declared Doppler band')
        return result


def crossvalidated_metric(residual, dt_s, widths=(.25,.5,1.,2.,4.,8.,16.,64.), *, scalar=False):
    """Select spectral resolution by excluded Fourier-sample likelihood.

    Alternating Fourier vectors are excluded from each covariance estimate.
    All coordinates are observed residuals, with no injected trajectory or
    evaluation truth. Independence of these vectors is a model assumption.
    """
    n,r = residual.shape
    spectrum = np.fft.fft(residual,axis=0,norm='ortho')
    rank = np.argsort(np.argsort(np.fft.fftfreq(n,dt_s)))
    candidates = []
    for width in widths:
        bands = min(n//4,max(1,int(round(1/dt_s/width))))
        labels = rank*bands//n
        score=0.
        for b in range(bands):
            indices=np.flatnonzero(labels==b)
            for parity in range(2):
                train,test=indices[parity::2],indices[1-parity::2]
                cov,_=sample_covariance(spectrum[train])
                if scalar: cov=np.eye(r)*np.trace(cov).real/r
                _,logdet=np.linalg.slogdet(cov)
                z=spectrum[test]
                score+=len(test)*logdet+np.sum(z.conj()*np.linalg.solve(cov,z.T).T).real
        candidates.append(dict(width_hz=width,bands=bands,labels=labels,predictive_nll_per_element=score/(n*r)))
    selected=min(candidates,key=lambda q:q['predictive_nll_per_element'])
    labels=selected['labels'];covariance=[];amounts=[]
    for b in range(selected['bands']):
        cov,amount=sample_covariance(spectrum[labels==b])
        if scalar: cov=np.eye(r)*np.trace(cov).real/r
        covariance.append(cov);amounts.append(amount)
    result=FourierMetric(labels,covariance)
    result.description=dict(selected_width_hz=selected['width_hz'],scalar=scalar,
        crossvalidation=[{k:v for k,v in c.items() if k!='labels'} for c in candidates],
        shrinkage=amounts,eigenvalues=np.linalg.eigvalsh(covariance).tolist(),
        assumption='Block-circulant covariance; zero-mean independent Fourier vectors; alternating-frequency predictive covariance selection')
    return result


class MetricExperiment(Experiment):
    """The same Cartesian forward model under a frozen training covariance."""
    def __init__(self, base, metrics):
        self.__dict__.update(base.__dict__)
        if len(metrics) != len(self.fields):
            raise ValueError('One metric is required for each observed band')
        self.metrics = metrics
        self.q = []
        for f, train, design, metric in zip(self.fields, self.training, self.designs, metrics):
            use = np.repeat(train, len(f.offsets))
            d = metric.apply(design[use])
            u, s, _ = np.linalg.svd(d, full_matrices=False)
            self.q.append(u[:, s > s[0]*1e-10] if len(s) else u)

    def profile(self, p, y, *, derivative=True):
        residuals, jacobians, coefficients, predictions = [], [], [], []
        for c, (f, train, q, metric) in enumerate(zip(self.fields, self.training, self.q, self.metrics)):
            value = self.templates(p, c, derivatives=derivative)
            h, dh = value if derivative else (value, None)
            use = np.repeat(train, len(f.offsets))
            h_training = metric.apply(h[use])
            data = metric.apply(y[c][use])
            a = h_training-q@(q.conj().T@h_training)
            clean = data-q@(q.conj().T@data)
            coefficient = np.linalg.lstsq(a, clean, rcond=1e-10)[0]
            r = clean-a@coefficient
            if derivative:
                da = metric.apply(dh[use])
                da -= np.einsum('ni,ijq->njq', q, np.einsum('ni,njq->ijq', q.conj(), da))
                v = np.einsum('nsq,s->nq', da, coefficient)
                inverse = np.linalg.pinv(a, rcond=1e-10)
                dc = -inverse@v+inverse@inverse.conj().T@np.einsum('nsq,n->sq', da.conj(), r)
                dr = -v-a@dc
                jacobians.append(np.r_[dr.real, dr.imag])
            residuals.append(np.r_[r.real, r.imag]); coefficients.append(coefficient)
            static = np.linalg.lstsq(metric.apply(self.designs[c][use]),
                data-h_training@coefficient, rcond=1e-10)[0]
            predictions.append(h@coefficient+self.designs[c]@static)
        return np.concatenate(residuals), np.concatenate(jacobians) if derivative else None, coefficients, predictions
