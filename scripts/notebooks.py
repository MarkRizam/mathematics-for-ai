"""Generate and execute compact instructional notebooks."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient

LESSONS = [
('01-linear-algebra','Vectors and matrices','A matrix transforms a vector; verify the dot product and transformation.','x=np.array([1.,2.]); A=np.array([[2.,0.],[1.,3.]])\nprint(A@x, x@x)\nassert np.allclose(A@x,[2,7])','Change one matrix entry and predict the resulting vector.'),
('02-lu','Pivoted LU','Factor P A = L U and solve a system using triangular substitutions.','A=np.array([[0.,2.],[1.,3.]])\nP,L,U=lu(A)\nassert np.allclose(P@A,L@U)\nprint(solve(A,[2,4]))','Try a singular matrix and explain why a pivot check is necessary.'),
('03-qr','QR factorization','Modified Gram-Schmidt constructs orthogonal columns.','A=np.array([[1.,2.],[2.,0.],[3.,1.]])\nQ,R=qr(A)\nprint(Q.T@Q)\nassert np.allclose(Q@R,A)','Compare orthogonality on nearly dependent columns.'),
('04-svd-eigen','Eigenpairs and SVD','Use NumPy reference solvers; low rank reconstruction discards smaller singular values.','A=np.array([[3.,1.],[1.,3.]])\nw,V=np.linalg.eigh(A)\nU,s,VT=np.linalg.svd(A)\nprint(w,s)\nassert np.allclose(A@V,V*w)\nprint(np.linalg.norm(A-(U[:,:1]*s[:1])@VT[:1]))','Compare rank-one and full-rank errors.'),
('05-gradients','Numerical differentiation','Central differences trade truncation error against roundoff.','x=np.array([.2,.8])\nfor h in [1e-1,1e-5,1e-9]: print(h,np.linalg.norm(gradient(lambda t:np.sin(t).sum(),x,h)-np.cos(x)))','Try additional step sizes; find the smallest observed error.'),
('06-optimization','Optimization','Compare GD, momentum and Adam at fixed iteration counts.','for method in ["gd","momentum","adam"]:\n    x,h=optimize(lambda x:x@x,lambda x:2*x,[2.,-1.],method=method,rate=.03,steps=400)\n    print(method,h[-1])\n    assert h[-1]<1e-6','Change the rate; equal iterations do not guarantee equal work.'),
('07-autodiff','Reverse-mode autodiff','Accumulate derivatives through shared graph nodes using the chain rule.','x,y=Value(2),Value(3)\nz=x*y+x*x\nz.backward()\nprint(x.grad,y.grad)\nassert (x.grad,y.grad)==(7,2)','Compare against a finite difference or analytic gradient.'),
]

def main():
    folder=Path('notebooks'); folder.mkdir(exist_ok=True)
    for name,title,theory,code,exercise in LESSONS:
        book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell('# '+title+'\n\n'+theory),nb.v4.new_code_cell('import numpy as np\nfrom math_for_ai import Value, gradient, lu, optimize, qr, solve\n'+code),nb.v4.new_markdown_cell('**Exercise:** '+exercise+'\n\nReference: https://numpy.org/doc/stable/reference/routines.linalg.html\n\nEducational examples only.')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
        NotebookClient(book,timeout=60,kernel_name='python3').execute()
        nb.write(book,folder/(name+'.ipynb'))
        print(name,'PASS',flush=True)

if __name__=='__main__': main()
