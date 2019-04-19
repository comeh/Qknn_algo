addprocs(6);

@everywhere using Distributions
@everywhere using Optim

c=10
@everywhere function truc(c)
    N=101; T=1; H=T/(N-1)
    Y0=0; sigma=.1; kappa=.5; rho=.5
    X0=10 ########### ICI on assimilie Xbarre à X
    sqrth=sqrt(H);

    eta=100
    c=c

    r(y,a)= (1/2*a*a+eta/2*y)*H #instantaneous reward
    g(y)=c/2*y # Terminal reward

    Nw=51
    readfile=readdlm("one_dim_1_1000/$(Nw)_1_nopti")
    Gridw=readfile[:,2]
    pop!(Gridw)
    GridW=[sqrt(t/(N-1))*Gridw for t=0:N-1];
    Ny=601
    GridY=[]
    push!(GridY,[0])
    for t=0:N-2
        Gridy=readdlm("one_dim_1_1000/$(Ny)_1_nopti")[:,2] ### Rajouter+t
        #println(Gridy)
        pop!(Gridy)
        Gridy=.1+1/16*Gridy # *sqrt(t/(N-1))
        Gridy=Gridy[Gridy.>0]
        push!(GridY,Gridy);
    end
    #lenGridy=length(GridY[1])
    #lenGridw= length(Gridw)

    function projy(n,eps,w,y,a) #Return the projection of ynext on GridY[n+1] # eps= W[n+1]-W[n]
        x=X0*exp(sigma*rho*w-1/2*sigma^2*rho^2*T/(N-1)*(n-1))
        ynext = (y==0) ? sigma^2*(1-rho^2)*x^2*H : y+((sigma^2-2*(kappa+a))*y+sigma^2*(1-rho^2)*x^2)*H+2*rho*sigma*y*eps
        #println("ynext ", ynext)
        if ynext>= GridY[n+1][length(GridY[n+1])]
            return (GridY[n+1][length(GridY[n+1])],GridY[n+1][length(GridY[n+1])])
        elseif ynext <= GridY[n+1][1]
            #println(ynext)
            return (GridY[n+1][1],GridY[n+1][1])
        else
            int = searchsortedfirst(GridY[n+1],ynext)
            return (GridY[n+1][int-1],GridY[n+1][int])
        end
    end

    function projw(n,eps,w) #Return the projection of wnext on GridW[n+1] # eps= W[n+1]-W[n]
        wnext=w+eps
        #println("wnext ", wnext)
        if wnext>= GridW[n+1][length(GridW[n+1])]
            return GridW[n+1][length(GridW[n+1])]
        elseif wnext <= GridW[n+1][1]
            #println(wnext)
            return GridW[n+1][1]
        else
            int = searchsortedfirst(GridW[n+1],wnext)
            if wnext-GridW[n+1][int-1] < GridW[n+1][int]- wnext
                return GridW[n+1][int-1]
            else
                return GridW[n+1][int]
            end
        end
    end

    Strat=[Dict() for t in 0:N-1]; # Strategy
    ValueFunction=[Dict() for t in 0:N-1]; # Strategy
    V=Dict(); # Value function  ########################## Coder la fonction valeur comme une map
    for w in GridW[N]
        for y in GridY[N]
            V[(w,y)]=g(y)
        end
    end

    function phi0(x) # return the cdf of the Normal(0,1) law
        return cdf(Normal(),x)
    end

    Nnorm=51# number of points for the quantization of w
    GridN=readdlm("one_dim_1_1000/$(Nnorm)_1_nopti")[:,2]
    pop!(GridN)

    function expectation(n, w, y, V, a)
        x=X0*exp(sigma*rho*w-1/2*sigma^2*rho^2*T/(N-1)*(n-1))
        ynext = (y==0) ? sigma^2*(1-rho^2)*x^2*H : y+((sigma^2-2*(kappa+a))*y+sigma^2*(1-rho^2)*x^2)*H+2*rho*sigma*y*(sqrth*GridN[1])
        ym,yp=projy(n,sqrth*GridN[1],w,y,a)
        lambda= yp>ym ? (ynext-ym)/(yp-ym):1
        wn=projw(n,sqrth*GridN[1],w)
        res=phi0((GridN[1]+GridN[2])/2)*(lambda*V[(wn,yp)]+(1-lambda)*V[(wn,ym)])
        for i=2:Nnorm-1
            ynext = (y==0) ? sigma^2*(1-rho^2)*x^2*H : y+((sigma^2-2*(kappa+a))*y+sigma^2*(1-rho^2)*x^2)*H+2*rho*sigma*y*sqrth*GridN[i]
            ym,yp=projy(n,sqrth*GridN[i],w,y,a)
            lambda= yp>ym ? (ynext-ym)/(yp-ym):1
            wn=projw(n,sqrth*GridN[i],w)
            res += (phi0((GridN[i]+GridN[i+1])/2)-phi0((GridN[i]+GridN[i-1])/2))*(lambda*V[(wn,yp)]+(1-lambda)*V[(wn,ym)])
        end
        ynext = (y==0) ? sigma^2*(1-rho^2)*x^2*H : y+((sigma^2-2*(kappa+a))*y+sigma^2*(1-rho^2)*x^2)*H+2*rho*sigma*y*GridN[Nnorm]
        ym,yp=projy(n,GridN[Nnorm],w,y,a)
        lambda= yp>ym ? (ynext-ym)/(yp-ym):1
        wn=projw(n,sqrth*GridN[Nnorm],w)
        res+=(1-phi0((GridN[Nnorm]+GridN[Nnorm-1])/2))*(lambda*V[(wn,yp)]+(1-lambda)*V[(wn,ym)])
        return res+ r(y,a)
    end

    function optimalExpectation(n,w,y,V) # Return the minimum and the argmin
        ######################### changer l'intervalle de confiance avec un truc qui dépend du pas de grille en y
        ftemp(a)=expectation(n, w, y, V, a)
        res=optimize(ftemp,0,100.0,Brent())#Brent())#,abs_tol=1.)
        return (Optim.minimizer(res),Optim.minimum(res))
    end

    function backward(n,V)
        Vback=Dict()
        for w in [0]#GridW[n]
            for y in GridY[n]
                temp=optimalExpectation(n,w,y,V)
                Vback[(w,y)]=temp[2]
                ValueFunction[n][(w,y)]=temp[2]
                Strat[n][(w,y)]=temp[1]
            end
        end
        return Vback
    end

    Vback=V
    for t=N-1:-1:N-1
        println(t)
        Vback=backward(t,Vback)
    end

    function Valuetheo(y,a) #w=0
        x=X0*exp(-1/2*sigma^2*rho^2*(N-2)*H)
        return (a^2/2+eta/2*y)*H+c/2*(y+((sigma^2-2*(kappa+a))*y+sigma^2*(1-rho^2)*x^2)*H)
    end

    function valuetheo(y)
        ftemp(a)=Valuetheo(y,a)
        res=optimize(ftemp,0,1000.0,GoldenSection())#Brent())#,abs_tol=1.)
        return Optim.minimum(res)
    end

    function strattheo(y)
        ftemp(a)=Valuetheo(y,a)
        res=optimize(ftemp,0,1000.0,GoldenSection())#Brent())#,abs_tol=1.)
        return Optim.minimizer(res)
    end

    using Plots
    gr()
    Xabs=GridY[N-1]
    scatter([[Strat[N-1][(0,x)] for x in Xabs] [strattheo(y) for y in Xabs]])


    function projeteY(n,y) #Return the projection of y+a*H on GridW[n]
        if y>= GridY[n][length(GridY[n])]
            return GridY[n][length(GridY[n])]
        elseif y <= GridY[n][1]
            return GridY[n][1]
        else
            int = searchsortedfirst(GridY[n],y)
            if abs(y-GridY[n][int]) < abs(y-GridY[n][int-1])
                return GridY[n][int]
            else
                return GridY[n][int-1]
            end
        end
    end

    function projeteW(w,n)
        if w>= GridW[n][length(GridW[n])]
            return GridW[n][length(GridW[n])]
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
        X=[X0 for i in 1:NbTirages]
        W=[0 for i in 1:NbTirages]
        Yquant=[Y0 for i in 1:NbTirages]
        J=[0 for i in 1:NbTirages]
        aopt=[Strat[1][(0,projeteY(1,y))] for y in Yquant]
        for n in 2:N
            J+=[r(Yquant[i],aopt[i]) for i in 1:NbTirages]
            #println("valeur ", J[1], " control ", aopt[1])
            dW=rand(dN,NbTirages)
            Yquant+=[((sigma^2-2*(kappa+aopt[i]))*Yquant[i]+sigma^2*(1-rho^2)*X[i]^2)*H+2*rho*sigma*Yquant[i]*sqrth*dW[i] for i in 1:NbTirages]
            #println("yquant ",Yquant[1])
            #println("bornes ", minimum(GridY[n]), " " , maximum(GridY[n]))
            X+=[sigma*rho*X[ind]*sqrth*dW[ind] for ind in 1:NbTirages]
            #println("X ", X[1])
            W+=sqrth*dW
            if n<N
                aopt=[Strat[n][(projeteW(W[ind],n),projeteY(n,Yquant[ind]))] for ind in 1:NbTirages]
            end
        end
        J+=[g(y) for y in Yquant]
        #println(J[N])
        #println(Yquant)
        res=sum(J)/NbTirages
        return res
    end

    N=100
    H=T/(N-1)
    sqrth=sqrt(H)
    function benchmark(NbTirages)
        X=[X0 for i in 1:NbTirages]
        W=[0 for i in 1:NbTirages]
        Y=[Y0 for i in 1:NbTirages]
        J=[0 for i in 1:NbTirages]
        abench=[0 for y in Y]
        for n in 2:N
            J+=[r(Y[i],abench[i]) for i in 1:NbTirages]
            dW=sqrth*rand(dN,NbTirages)
            dW=ones(NbTirages,1)
            Y+=[((sigma^2-2*(kappa+abench[i]))*Y[i]+sigma^2*(1-rho^2)*X[i]^2)*H+2*rho*sigma*Y[i]*dW[i] for i in 1:NbTirages]
            println(n, " " , Y)
            X+=[sigma*rho*X[ind]*dW[ind] for ind in 1:NbTirages]
            #println(X[1])
            W+=sqrth*dW

        end
        J+=[g(y) for y in Y]
        #println(Y)
        res=mean(J)
        return res
    end

    xi=1
    sigma^2*(1-rho^2)*X0.^2*H
    H=0.01
    Y0
    benchmark(1)
    return ValueQuantif(100000)
end

c=[0.,1.,5.,10.,25.,50.]
a=pmap(truc,c)

writedlm("resultsrhosmooth.txt", a)
