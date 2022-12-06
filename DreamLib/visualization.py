import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def explo_DR(df,Y):
    print(Y.describe())
    fig, axes = plt.subplots(4,figsize=(15,30))

    sns.lineplot(x=df.index, y=Y.values,color= 'r',ax=axes[0])

    sns.histplot(data=Y, bins = 20, color='c', edgecolor='k',ax=axes[1])


    (mu, sigma) = stats.norm.fit(Y)
    sns.distplot(Y, fit=stats.norm, ax=axes[2])
    axes[2].legend(['Kernel density estimation','Normal dist. ($\mu=$ {:.8f} and $\sigma=$ {:.8f} )'.format(mu, sigma)])

    stats.probplot(Y, plot=axes[3])
    plt.show()
    

def explo_portefeuille(df,Y,num):
    *cols, = map(lambda x: x+'_'+str(num), ['mean', 'median', 'p5', 'p10', 'p25', 'p75', 'p90','p95'])

    fig, ax1 = plt.subplots(figsize=(25,10))
    for i in range(8):
        ax1.plot(df.index, df[cols[i]].values)
        ax2 = ax1.twinx()
        ax2.plot(df.index, Y.values, color='k', linewidth=2)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True,ncol=2, labels=cols)
    plt.show()

    fig, axes = plt.subplots(8,2,figsize=(25,15))
    for i, col in enumerate(cols):
        sns.regplot(x=df[col].values,y=Y.values, ax=axes[i,0])
        sns.residplot(x=df[col],y=Y.values, ax=axes[i,1])
        axes[i,0].title.set_text(col)
        axes[i,1].title.set_text(col)
    plt.show()

def explo_client(df,Y):
    d1 = df['CD_TY_CLI_RCI_1'].values
    d2 = df['CD_TY_CLI_RCI_2'].values
    colors = sns.color_palette(palette="pastel", n_colors = 2, )

    fig, ax1 = plt.subplots()
    ax1.stackplot(df.index, d1, d2, colors =colors)
    half_size_band = 0.01
    ax1.set_ylim([max(d1) - half_size_band, max(d1) + half_size_band])

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='k', linewidth=2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True,ncol=2, labels=['Particulier','Personne physique'])
    plt.show()

    fig, axes = plt.subplots(2)
    sns.regplot(x=df['CD_TY_CLI_RCI_1'].values,y=Y.values, ax=axes[0])
    sns.residplot(x=df['CD_TY_CLI_RCI_1'].values,y=Y.values, ax=axes[1])
    plt.show()
    

def explo_habitation(df,Y):
    d1 = df['CD_MOD_HABI_1'].values
    d2 = df['CD_MOD_HABI_2'].values
    colors = sns.color_palette(palette="pastel", n_colors = 2, )

    fig, ax1 = plt.subplots()
    ax1.stackplot(df.index, d1, d2, colors =colors)
    half_size_band = 0.05
    ax1.set_ylim([max(d1) - half_size_band, max(d1) + half_size_band])
    ax1.set_ylabel('Housing type repartition')

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='k', linewidth=2)
    ax2.set_ylabel('DR')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True,ncol=2, labels=['Tenants and others','Owners and missing'])
    plt.show()

    fig, axes = plt.subplots(2)
    sns.regplot(x=df['CD_MOD_HABI_1'].values,y=Y.values, ax=axes[0])
    axes[0].set_title('Reg DR and Housing repartition')
    sns.residplot(x=df['CD_MOD_HABI_1'].values,y=Y.values, ax=axes[1])
    axes[1].set_title('Residuals DR and Housing repartition')
    plt.show()


def explo_civil(df,Y):
    d1 = df['CD_ETA_CIV_1'].values
    d2 = df['CD_ETA_CIV_2'].values
    colors = sns.color_palette(palette="pastel", n_colors = 2, )

    fig, ax1 = plt.subplots()
    ax1.stackplot(df.index, d1, d2, colors =colors)
    half_size_band = 0.06
    ax1.set_ylim([max(d1) - half_size_band, max(d1) + half_size_band])

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='k', linewidth=2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True,ncol=2, labels=['Célibataires et Autres','Marié(e) et Manquant'])
    plt.show()

    fig, axes = plt.subplots(2)
    sns.regplot(x=df['CD_ETA_CIV_1'].values,y=Y.values, ax=axes[0])
    sns.residplot(x=df['CD_ETA_CIV_1'].values,y=Y.values, ax=axes[1])
    plt.show()


def explo_code_qualite(df,Y):
    d1 = df['CD_QUAL_VEH_1'].values
    d2 = df['CD_QUAL_VEH_2'].values
    colors = sns.color_palette(palette="pastel", n_colors = 2, )

    fig, ax1 = plt.subplots()
    ax1.stackplot(df.index, d1, d2, colors =colors)
    half_size_band = 0.1
    ax1.set_ylim([max(d1) - half_size_band, min(d2) + half_size_band])

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='k', linewidth=2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True,ncol=2, labels=['Véhicule d\'occasion','Véhicule neuf et Manquant'])
    plt.show()

    fig, axes = plt.subplots(2)
    sns.regplot(x=df['CD_QUAL_VEH_1'].values,y=Y.values, ax=axes[0])
    sns.residplot(x=df['CD_QUAL_VEH_1'].values,y=Y.values, ax=axes[1])
    plt.show()
    

def explo_profession(df,Y):
    d1 = df['CD_PROF_1'].values
    d2 = df['CD_PROF_2'].values
    d3 = df['CD_PROF_3'].values
    colors = sns.color_palette(palette="pastel", n_colors = 3, )

    fig, ax1 = plt.subplots()
    ax1.stackplot(df.index, d1, d2, d3, colors =colors)
    half_size_band = 0.15
    ax1.set_ylim([max(d1) - half_size_band, max(d2) + half_size_band])

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='k', linewidth=2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True,ncol=2, 
            labels=['Ouvriers et Autres','Employés, Autres personnels de servies','Retraites, Cadres, Professions libérales, Ingénieurs, Agents de Maitrise, ... et Manquants'])
    plt.show()

    fig, axes = plt.subplots(2,2)
    sns.regplot(x=df['CD_QUAL_VEH_1'].values,y=Y.values, ax=axes[0,0])
    sns.residplot(x=df['CD_QUAL_VEH_1'].values,y=Y.values, ax=axes[0,1])
    sns.regplot(x=df['CD_QUAL_VEH_2'].values,y=Y.values, ax=axes[1,0])
    sns.residplot(x=df['CD_QUAL_VEH_2'].values,y=Y.values, ax=axes[1,1])
    plt.show()
    

def explo_pib(df,Y):
    fig, ax1 = plt.subplots()
    ax1.plot(df.index, df['PIB'].values, color='k', linewidth=2)

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='r', linewidth=2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.4, -0.06),
            fancybox=True, shadow=True,ncol=2, 
            labels=['PIB'])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.6, -0.06),
            fancybox=True, shadow=True,ncol=2, 
            labels=['DR'])
    plt.show()

    fig, axes = plt.subplots(2)
    sns.regplot(x=df['PIB'].values,y=Y.values, ax=axes[0])
    sns.residplot(x=df['PIB'].values,y=Y.values, ax=axes[1])
    plt.show()


def explo_inflation(df,Y):
    fig, ax1 = plt.subplots()
    ax1.plot(df.index, df['Inflation'].values, color='k', linewidth=2)

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='r', linewidth=2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.4, -0.06),
            fancybox=True, shadow=True,ncol=2, 
            labels=['Inflation'])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.6, -0.06),
            fancybox=True, shadow=True,ncol=2, 
            labels=['DR'])
    plt.show()

    fig, axes = plt.subplots(2)
    sns.regplot(x=df['Inflation'].values,y=Y.values, ax=axes[0])
    sns.residplot(x=df['Inflation'].values,y=Y.values, ax=axes[1])
    plt.show()


def explo_txcho(df,Y):
    fig, ax1 = plt.subplots()
    ax1.plot(df.index, df['Tx_cho'].values, color='k', linewidth=2)

    ax2 = ax1.twinx()
    ax2.plot(df.index, Y.values, color='r', linewidth=2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.4, -0.06),
            fancybox=True, shadow=True,ncol=2, 
            labels=['Taux de chomage'])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.6, -0.06),
            fancybox=True, shadow=True,ncol=2, 
            labels=['DR'])
    plt.show()

    fig, axes = plt.subplots(2)
    sns.regplot(x=df['Tx_cho'].values,y=Y.values, ax=axes[0])
    sns.residplot(x=df['Tx_cho'].values,y=Y.values, ax=axes[1])
    plt.show()