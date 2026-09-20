# Tutoriel : des vecteurs libres aux structures contraintes

L'idée essentielle est de séparer deux représentations d'un même paramètre :

- le côté **non contraint** est un vecteur dans ``\mathbb R^d``, pratique pour
  un optimiseur ou un algorithme MCMC ;
- le côté **contraint** est un objet Julia lisible dont les champs respectent
  automatiquement leurs contraintes.

TransformVariables.jl fournit les transformations mathématiques. Paramorph
ajoute une macro qui les associe aux champs d'une structure.

## 1. TransformVariables.jl seul

Une transformation lit un ou plusieurs réels libres et produit une valeur
contrainte :

```@example tutorial
using TransformVariables

t = asℝ₊
dimension(t)
transform(t, [0.0])
inverse(t, 1.0)
```

Ici, `asℝ₊` est l'exponentielle : `0.0` devient `1.0`. Les quatre opérations à
retenir sont :

- `dimension(t)` : nombre de coordonnées libres consommées ;
- `transform(t, x)` : passage du vecteur libre à la valeur contrainte ;
- `inverse(t, y)` : retour vers le vecteur libre ;
- `transform_and_logjac(t, x)` : transformation et log-déterminant du
  Jacobien, utile en calcul probabiliste.

Les transformations se composent :

```@example tutorial
t = as((
    position = asℝ,
    scale = asℝ₊,
    weights = UnitSimplex(3),
))

dimension(t) # 1 + 1 + (3 - 1)
y = transform(t, zeros(dimension(t)))
inverse(t, y)
```

Un simplexe de longueur `N` n'a que `N - 1` degrés de liberté : ses
composantes sont positives et leur somme vaut un.

## 2. Une première structure contrainte

Avec Paramorph, la transformation suit le type du champ dans une seconde
annotation `::` :

```@example tutorial
using Paramorph

@constrained_struct struct MixtureParameters{T, N}
    location::T::asℝ
    scale::T::asℝ₊
    weights::Vector{T}::UnitSimplex(N)
end

P = MixtureParameters{Float64, 3}
dimension_intrinsique(P)
```

La macro conserve une structure Julia ordinaire et crée en plus
`transformation_schema(P)`, la transformation qui sait construire un `P`. Le
paramètre de type `N` détermine ici la taille du simplexe.

```@example tutorial
x = zeros(dimension_intrinsique(P))
p = constraint(P, x)

typeof(p)
p.scale
p.weights
unconstrain(p) ≈ x
```

Le nom `constraint` signifie ici « appliquer la transformation ». Il reçoit un
**type** et un vecteur. `unconstrain` reçoit l'**objet** et retrouve son vecteur.

## 3. Imbriquer des structures

Un champ sans seconde annotation est considéré comme une structure possédant
déjà son propre `transformation_schema` :

```@example tutorial
@constrained_struct struct ModelParameters{T, N}
    mixture::MixtureParameters{T, N}
    negative_offset::T::asℝ₋
end

M = ModelParameters{Float64, 3}
x = zeros(dimension_intrinsique(M))
m = constraint(M, x)

(m.mixture.scale, m.mixture.weights, m.negative_offset)
unconstrain(m) ≈ x
```

La dimension totale est la somme récursive des dimensions des champs.

## 4. Facteur de Cholesky d'une corrélation

Dans TransformVariables 0.8, les constructeurs sont `UnitSimplex(N)` et
`corr_cholesky_factor(N)`. Le second produit un `UpperTriangular`, pas une
matrice de corrélation : si `U` est le résultat, la corrélation vaut `U' * U`.

```@example tutorial
using LinearAlgebra

@constrained_struct struct CorrelationParameters{T, N}
    U::UpperTriangular{T, Matrix{T}}::corr_cholesky_factor(N)
end

C = CorrelationParameters{Float64, 3}
c = constraint(C, zeros(dimension_intrinsique(C)))
R = c.U' * c.U
diag(R)
```

Sa dimension libre est `N * (N - 1) / 2`. Le type du champ doit accepter le
résultat de la transformation : `Matrix{T}` seul n'accepterait pas
`UpperTriangular{T, Matrix{T}}`.

## 5. Le log-Jacobien

Une densité sur les paramètres contraints doit être corrigée lorsqu'on
l'évalue dans les coordonnées libres :

```@example tutorial
x = randn(dimension_intrinsique(M))
m, logjac = constraint_with_logjac(M, x)

isfinite(logjac)
unconstrain(m) ≈ x
```

Si `logdensity_constrained(m)` est une log-densité exprimée sur l'objet
contraint, on écrit schématiquement :

```julia
m, logjac = constraint_with_logjac(M, x)
logdensity_unconstrained = logdensity_constrained(m) + logjac
```

Ne rajoutez pas cette correction si la bibliothèque statistique appelée la
prend déjà en charge.

## 6. Aide-mémoire

| Valeur souhaitée | Transformation | Dimension libre |
|:--|:--|:--|
| réel | `asℝ` | 1 |
| réel strictement positif | `asℝ₊` | 1 |
| réel strictement négatif | `asℝ₋` | 1 |
| réel entre 0 et 1 | `as𝕀` | 1 |
| vecteur de `N` réels | `as(Vector, asℝ, N)` | `N` |
| simplexe de longueur `N` | `UnitSimplex(N)` | `N - 1` |
| facteur de corrélation `N × N` | `corr_cholesky_factor(N)` | `N(N - 1)/2` |

`methods(as)` montre les constructeurs génériques et l'aide Julia, par exemple
`?UnitSimplex`, documente chaque transformation.

## 7. Erreurs fréquentes

- Le vecteur passé à `constraint` doit avoir exactement
  `dimension_intrinsique(T)` éléments.
- La transformation doit produire une valeur compatible avec le type du champ.
- Sans transformation explicite, Paramorph appelle
  `transformation_schema(TypeDuChamp)`. Le défaut pour un scalaire est `asℝ` ;
  une structure imbriquée doit avoir été déclarée avec `@constrained_struct`.
- `asSimplex` et `asCorrelationCholesky` ne font pas partie de
  TransformVariables 0.8. Utilisez `UnitSimplex` et
  `corr_cholesky_factor`.
