
Sigma=.2 # volatility of the asset
T=1 # terminal time
Gamma0=.5; B0=.3 # parameters in the prior law of the drift Beta
w0=0; s0=6; y0= 1; # initialization of the processes W, S, Y
P=5 # parameter in the CARA utility function
N=100; # subdivision of [0,T]
H=T/(N-1) # pas de subdivision en temps
sqrth=sqrt(H)

Gamma=5 # control penalization
Eta=10 # inventory penalization 

function f(a)
    return Gamma*a
end

function l(y)
    return Eta*y*y
end

F(t, w) = exp(1./2*Sigma*Sigma*Gamma0*Gamma0/(Sigma*Sigma+Gamma0*Gamma0*t)*(w/Sigma+B0/(Gamma0*Gamma0))*(w/Sigma+B0/(Gamma0*Gamma0))- B0*B0/(2*Gamma0*Gamma0));

## Instantaneous reward
function r(n,w,y,a)
    s=s0*exp(Sigma*w-Sigma^2/2*H*(n-1))
    return F(H*(n-1),w)*a*(s+f(a))*H
end

## Final reward
function g(w,y)
    return F(T,w)*l(y)
end

#Stupid Grid N(0,1)
Nw=101 # number of points for the quantization of w
readfile=readdlm("one_dim_1_1000/$(Nw)_1_nopti")
Gridw=readfile[:,2]
pop!(Gridw)
GridW=[sqrt(t/(N-1))*Gridw for t=0:N-1];
Ny=201 # number of points for the quantization of y
Ymin=-0.5
Ymax=1
Gridy=[Ymin+(Ymax-Ymin)*i/(Ny-1) for i=0:Ny-1];
lenGridy= length(Gridy)


function projy(y,a) #Return the projection of y+a*H on GridY
    if y+a*H>= Gridy[lenGridy]
        return Gridy[lenGridy]
    elseif y+a*H <= Gridy[1]
        return Gridy[1]
    else
        int = searchsortedfirst(Gridy,y+a*H)
        if abs(y+a*H-Gridy[int]) < abs(y+a*H-Gridy[int-1])
            return Gridy[int]
        else
            return Gridy[int-1]
        end
    end
end

Strat=[Dict() for t in 0:N-1] # Strategy
ValueFunction=[Dict() for t in 0:N-1] # Strategy
V=Dict() # Value function  ########################## Coder la fonction valeur comme une map
for w in Gridw
    for y in Gridy
        V[(w,y)]=F(T,w)*l(y)
    end
end

using Distributions
function phi0(x) # return the cdf of the Normal(0,1) law
    return cdf(Normal(),x)
end

sqrth=sqrt(H)
function expectation(n, w, y, V, a) 
    yproj=projy(y,a)
    #println(y+a*H, " projeté sur ", yproj)
    res=phi0(((GridW[n+1][1]+GridW[n+1][2])/2-w)/sqrth)*V[(GridW[n+1][1],yproj)] 
    for i=2:Nw-1
        res += (phi0(((GridW[n+1][i]+GridW[n+1][i+1])/2-w)/sqrth)-phi0(((GridW[n+1][i]+GridW[n+1][i-1])/2-w)/sqrth))*V[(GridW[n+1][i],yproj)]
    end
    res+=(1-phi0(((GridW[n+1][Nw]+GridW[n+1][Nw-1])/2-w)/sqrth))*V[(GridW[n+1][Nw],yproj)]
    return res+ r(n,w,y,a)
end

using Optim
using ProfileView

function optimalExpectation(n,w,y,V) # Return the minimum and the argmin 
    ######################### changer l'intervalle de confiance avec un truc qui dépend du pas de grille en y
    ftemp(a)=expectation(n, w, y, V, a)
    res=optimize(ftemp,-50.0,10.0,GoldenSection())#,abs_tol=absTol*.5)
    return (Optim.minimizer(res),Optim.minimum(res))
end

function backward(n,V)
    Vback=Dict()
    for w in GridW[n]
        for y in Gridy
            temp=optimalExpectation(n,w,y,V)
            Vback[(w,y)]=temp[2]
            ValueFunction[n][(w,y)]=temp[2]
            Strat[n][(w,y)]=temp[1]
        end
    end
    return Vback
end            

Vback=V
for t=N-1:-1:1
    println(t)
    Vback=backward(t,Vback)
end

Beta=rand(Normal(B0,Gamma0^2))

ConditionalExpectation(n)= sqrt(pi)* e^(-B0^2/(4 * Gamma0^2/2))* (erfi((B0 + 2* Gamma0^2/2* (T-(n-1)*H))/(2* sqrt(Gamma0^2/2))) - erfi(B0/(2 *sqrt(Gamma0^2/2))))/(2 *sqrt(Gamma0^2/2))


dN=Normal(0,1)
S=s0
Yopt=y0
J=0
aopt=-y0/(T+Gamma/Eta)-s0/2/Gamma
for n in 2:N
    J+=aopt*(S+f(aopt))*H
    S+=S*(Beta*H+Sigma*sqrth*rand(dN))
    Yopt+=aopt*H
    if n<N
    aopt=-Yopt/(T-(n-1)*H+Gamma/Eta)+1/(2*Eta)*(1/(T-(n-1)*H+Gamma/Eta)*S*ConditionalExpectation(n)-S)
    end
end
J+=l(Yopt)
J

dN=Normal(0,1)
function ValueOpt(NbTirages)
    res=0
    for nb in 1:NbTirages
        S=s0
        Yopt=y0
        J=0
        aopt=-y0/(T+Gamma/Eta)-s0/2/Gamma
        for n in 2:N
            J+=aopt*(S+f(aopt))*H
            S+=S*(Beta*H+Sigma*sqrth*rand(dN))
            Yopt+=aopt*H
            if n<N
            aopt=-Yopt/(T-(n-1)*H+Gamma/Eta)+1/(2*Eta)*(1/(T-(n-1)*H+Gamma/Eta)*S*ConditionalExpectation(n)-S)
            end
        end
        #println(Yopt)
        J+=l(Yopt)
        res+=J/NbTirages
    end
    return res
end


## Naive Strategy

dN=Normal(0,1)
function ValueFoo(NbTirages)
    res=0
    for nb in 1:NbTirages
        S=s0
        Yopt=y0
        J=0
        aopt=-y0
        for n in 2:N
            J+=aopt*(S+f(aopt))*H
            S+=S*(Beta*H+Sigma*sqrth*rand(dN))
            Yopt+=aopt*H
        end
        J+=l(Yopt)
        res+=J/NbTirages
    end
    return res
end

## Quantized Strategy

lenGridy= length(Gridy)
lenGridw= length(Gridw)

function projeteY(y) #Return the projection of y+a*H on GridW[n]
    if y>= Gridy[lenGridy]
        return Gridy[lenGridy]
    elseif y <= Gridy[1]
        return Gridy[1]
    else
        int = searchsortedfirst(Gridy,y)
        if abs(y-Gridy[int]) < abs(y-Gridy[int-1])
            return Gridy[int]
        else
            return Gridy[int-1]
        end
    end
end

function projeteW(w,n)
    if w>= GridW[n][lenGridw]
        return GridW[n][lenGridw]
    elseif w <= GridW[n][1]
        return GridW[n][1]
    else
        int = searchsortedfirst(GridW[n],w)
        if abs(w-GridW[n][int]) < abs(w-GridW[n][int-1])
            return GridW[n][int]
        else
            return GridW[n][int-1]
        end
    end
end

dN=Normal(0,1)
function ValueQuantif(NbTirages)
    S=[s0 for i in 1:NbTirages]
    W=[0 for i in 1:NbTirages]
    Yquant=[y0 for i in 1:NbTirages]
    J=[0 for i in 1:NbTirages]
    aopt=[Strat[1][(0,projeteY(y))] for y in Yquant]
    for n in 2:N
        J+=diagm(aopt)*(S+f(aopt))*H
        dW=rand(dN,NbTirages)
        S+=[S[ind]*(Beta*H+Sigma*sqrth*dW[ind]) for ind in 1:NbTirages]
        W+=sqrth*dW
        Yquant+=aopt*H
        if n<N
        aopt=[Strat[n][(projeteW(W[ind],n),projeteY(Yquant[ind]))] for ind in 1:NbTirages]
        end    
    end
    J+=[l(y) for y in Yquant]
    #println(Yquant)
    res=sum(J)/NbTirages
    return res
end

writedlm("ValueFunction.txt",[Y0test VFFtest VFQtest VFOtest], " ; ")

function testgeneral(NbTirages)
    S=[s0 for i in 1:NbTirages]
    W=[0 for i in 1:NbTirages]
    Yquant=[y0 for i in 1:NbTirages]
    Yopt=[y0 for i in 1:NbTirages]
    Ybench=[y0 for i in 1:NbTirages]
    Jquant=[0 for i in 1:NbTirages]
    Jopt=[0 for i in 1:NbTirages]
    Jbench=[0 for i in 1:NbTirages]
    aQuant=[Strat[1][(0,projeteY(Yquant[i]))] for i in 1:NbTirages]
    aopt=[-y0/(T+Gamma/Eta)-s0/2/Gamma for i in 1:NbTirages]
    abench=[-y0/T for i in 1:NbTirages]
    for n in 2:N
        Jquant+=[aQuant[i]*(S[i]+f(aQuant[i]))*H for i in 1:NbTirages]
        Jopt+=[aopt[i]*(S[i]+f(aopt[i]))*H for i in 1:NbTirages]
        Jbench+=[abench[i]*(S[i]+f(abench[i]))*H for i in 1:NbTirages]
        dW=rand(dN,NbTirages)
        S+=[S[ind]*(Beta*H+Sigma*sqrth*dW[ind]) for ind in 1:NbTirages]
        W+=sqrth*dW
        Yquant+=aQuant*H
        Yopt+=aopt*H
        Ybench+=abench*H
        if n<N
            aQuant=[Strat[n][(projeteW(W[ind],n),projeteY(Yquant[ind]))] for ind in 1:NbTirages]
            aopt=[-Yopt[i]/(T-(n-1)*H+Gamma/Eta)+1/(2*Eta)*(1/(T-(n-1)*H+Gamma/Eta)*S[i]*ConditionalExpectation(n)-S[i]) for i in 1:NbTirages]
        end    
    end
    Jquant+=[l(y) for y in Yquant]
    Jopt+=[l(y) for y in Yopt]
    Jbench+=[l(y) for y in Ybench]
    #println(Yquant)
    EJquant=sum(Jquant)/NbTirages
    EJopt=sum(Jopt)/NbTirages
    EJbench=sum(Jbench)/NbTirages
    return EJquant, EJopt, EJbench
end

writedlm("ValueFunctiontest.txt",collect(testgeneral(10000)))
