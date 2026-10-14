# Plan de la API Python de QFOIL

## Estado y restricción de alcance

Este documento planifica la API Python del motor QFOIL existente. El objetivo
inmediato es ofrecer una superficie pública estable para parsear, evaluar,
buscar testigos y representar fórmulas QFOIL sin depender de una interfaz web.

La sintaxis del lenguaje declarativo descrita en
[`DECLARATIVE_MVP.md`](DECLARATIVE_MVP.md) todavía requiere aprobación. Por
ello, toda funcionalidad que dependa de esa sintaxis aparece al final de este
documento como **TODO bloqueado** y no forma parte de los hitos implementables
de este plan.

FastAPI, HTTP y KaTeX no son dependencias de esta biblioteca:

- QFOIL produce objetos Python y cadenas LaTeX compatibles con KaTeX.
- Un adaptador externo puede convertir esos objetos a JSON y exponerlos por
  HTTP.
- La interfaz visual es responsable de renderizar el LaTeX mediante KaTeX.

## Recorte para el proof of concept

La primera implementación demostrará el flujo con el menor número posible de
componentes. Este recorte tiene prioridad sobre las secciones más amplias del
plan:

1. una jerarquía mínima de errores públicos en las fronteras nuevas;
2. `WitnessSearchResult` y `QFOIL.search_witness()` como API aditiva;
3. `QFOILLatexRenderer` y `QFOIL.to_latex()` para el AST existente;
4. pruebas de contrato de esas operaciones y compatibilidad con la API actual.

Durante el proof of concept se posponen:

- una migración exhaustiva de todos los `ValueError` internos;
- un módulo público de alias de tipos si las anotaciones locales son
  suficientes;
- caché del LaTeX, plugins de formato y configuración visual;
- serialización del AST;
- cambios en las firmas de los métodos existentes;
- cualquier componente de `qfoil.queries`.

El proof of concept debe ser ampliable, pero no necesita resolver por
anticipado todos los contratos de una versión estable.

Estado del recorte PoC:

- [x] errores públicos mínimos;
- [x] `WitnessSearchResult` y `QFOIL.search_witness()`;
- [x] `QFOILLatexRenderer` y `QFOIL.to_latex()`;
- [x] pruebas focalizadas de contratos, validación y compatibilidad;
- [x] documentación de adopción en el README.

La suite completa, incluidas las integraciones PyJuice/CUDA, fue validada por
el mantenedor el 23 de septiembre de 2026. Con esta verificación se considera
cerrada la versión PoC de la API Python.

## Situación actual

La fachada `qfoil.QFOIL` ofrece ahora:

```python
qfoil.parse(expression)
qfoil.evaluate(expression_or_ast, bindings)
qfoil.define_function(name, parameters, body)
qfoil.grow_witness(expression, var, context, ...)
qfoil.search_witness(expression, var, context, ...)
qfoil.to_latex(expression_or_ast)
```

El recorte PoC ya agrega errores mínimos, resultado estructurado, conversión a
diccionario y representación LaTeX. Para convertir esta superficie en una API
estable todavía falta:

- extender las excepciones públicas a las fronteras antiguas cuando aporte
  valor;
- documentar de forma exhaustiva los tipos de entrada y salida;
- decidir si hacen falta alias públicos de tipos;
- validar el contrato de serialización con consumidores reales;
- documentación de compatibilidad y evolución de la API.

## Objetivos

1. Mantener `QFOIL` como fachada principal del paquete.
2. Estabilizar los contratos de parseo, evaluación y búsqueda de testigos.
3. Exponer errores que una aplicación pueda manejar sin inspeccionar mensajes
   de `ValueError`.
4. Generar LaTeX desde el AST QFOIL sin incorporar KaTeX ni dependencias web.
5. Ofrecer resultados estructurados y serializables sin acoplar la biblioteca
   a JSON o HTTP.
6. Conservar la independencia de los backends opcionales PyJuice y Z3.
7. Evitar decisiones públicas que condicionen la sintaxis declarativa pendiente
   de aprobación.

## No objetivos

Este plan no incluye:

- un servidor FastAPI;
- endpoints HTTP;
- autenticación, sesiones o persistencia;
- componentes visuales;
- renderizado HTML mediante KaTeX;
- el parser o executor del lenguaje declarativo;
- búsqueda conjunta de varias variables.

## Diseño de la API pública

### Fachada principal

La fachada continúa componiendo parser, runtime y backend probabilístico:

```python
from qfoil import QFOIL

qfoil = QFOIL(
    probabilistic_backend=backend,
    instance_len=8,
)
```

Las operaciones existentes permanecen disponibles:

```python
ast = qfoil.parse("SIZE(x) <= limit")

value = qfoil.evaluate(
    ast,
    bindings={"x": (1, None, 0), "limit": 2},
)

qfoil.define_function(
    "HAS_ONE",
    ["x"],
    "SIZE(x) == 1",
)

witness = qfoil.grow_witness(
    "FULL(z) * (z <<< x)",
    var="z",
    context={"x": (1, 0, 1)},
)
```

### Tipos compartidos

Se deben definir alias o protocolos públicos para evitar repetir contratos
informales:

```python
TupleValue = tuple[int | None, ...]
Bindings = Mapping[str, object]
Expression = str | ASTNode
```

Estos tipos documentan la API, pero no deben introducir dependencias en tiempo
de ejecución ni impedir que los backends acepten tipos numéricos compatibles
cuando corresponda.

### Jerarquía de errores

Se propone una jerarquía pequeña:

```python
class QFOILError(Exception): ...
class QFOILSyntaxError(QFOILError, ValueError): ...
class QFOILValidationError(QFOILError, ValueError): ...
class MissingBindingError(QFOILValidationError): ...
class BackendUnavailableError(QFOILError): ...
class UnsupportedOperationError(QFOILError): ...
```

La herencia desde `ValueError` en los errores de sintaxis y validación conserva
compatibilidad razonable con los consumidores actuales. Los mensajes deben
incluir la operación, el nombre involucrado y, cuando exista, la posición en el
texto fuente.

### Resultado estructurado de búsqueda

`grow_witness()` es parte de la API existente y debe conservar su retorno
`tuple | None`. Para consumidores que necesiten diagnóstico se añadirá una
operación aditiva, sin cambiar ese contrato:

```python
result = qfoil.search_witness(
    expression,
    var="candidate",
    context=bindings,
    budget=1000,
    pool_size=30,
    inner_budget=8,
    seed=None,
)
```

Resultado propuesto:

```python
WitnessSearchResult(
    status="found",              # found | not_found
    variable="candidate",
    witness=(None, 1, ...),
    verified=True,
    seed_used=None,
    search={
        "budget": 1000,
        "pool_size": 30,
        "inner_budget": 8,
    },
)
```

Reglas del contrato:

- `found` exige que el runtime vuelva a evaluar la fórmula con el testigo.
- `not_found` sólo indica agotamiento heurístico; no prueba inexistencia.
- `witness` es `None` cuando el estado es `not_found`.
- `grow_witness()` puede implementarse como una envoltura que extraiga
  `result.witness` para conservar compatibilidad.
- `WitnessSearchResult.to_dict()` debe producir sólo valores serializables por
  bibliotecas comunes, sin importar `json` ni imponer un transporte.

### Representación LaTeX de QFOIL

El impresor LaTeX pertenece al paquete del lenguaje porque conoce el AST y la
precedencia de QFOIL:

```text
qfoil/
  language/
    latex.py
```

API propuesta:

```python
latex = qfoil.to_latex("SIGMA z. FULL(z) * (z <<< x)")

# También acepta un AST ya parseado.
latex = qfoil.to_latex(ast)
```

La implementación pública de bajo nivel será:

```python
from qfoil.language import QFOILLatexRenderer

latex = QFOILLatexRenderer().render(ast)
```

El renderer debe:

- recorrer el AST, no transformar texto mediante expresiones regulares;
- preservar la precedencia mediante paréntesis cuando sean necesarios;
- escapar identificadores y no aceptar fragmentos LaTeX arbitrarios;
- generar un subconjunto compatible con KaTeX;
- representar `SIGMA`, `NOT`, `<<<`, `<<`, comparaciones, funciones y tuplas;
- usar `\cdot` para `*`, pues el AST QFOIL no contiene información suficiente
  para afirmar que toda multiplicación es una conjunción lógica;
- devolver solamente la expresión matemática, sin delimitadores `$...$`,
  HTML ni configuración de KaTeX.

Ejemplo esperado:

```python
qfoil.to_latex("SIGMA z. FULL(z) * (z <<< x)")
```

```latex
\sum_{z} \left(
  \operatorname{FULL}\left(z\right)
  \cdot \left(z \sqsubseteq x\right)
\right)
```

La generación es lineal respecto del tamaño del AST. Puede almacenarse en
caché por AST o texto normalizado si el perfil de uso demuestra que es
necesario; no se introducirá una caché antes de medir.

### Inspección y serialización

Los objetos de resultado deben ofrecer una descripción Python estable:

```python
result.to_dict()
```

No se agregará `to_json()`: la selección de biblioteca JSON, codificación y
transporte corresponde al consumidor. Los AST no necesitan serialización
pública en la primera versión; su contrato público es evaluarlos o
representarlos mediante la fachada.

## Organización propuesta

```text
qfoil/
  __init__.py
  api.py                 # fachada QFOIL
  errors.py              # excepciones públicas
  types.py               # alias y protocolos compartidos
  results.py             # WitnessSearchResult
  language/
    ast.py
    parser.py
    latex.py             # QFOILLatexRenderer
```

No se creará `qfoil/queries/` como parte de los hitos aprobados de este plan.
Ese paquete queda reservado para el trabajo declarativo bloqueado.

## Exportaciones públicas

El paquete raíz debe continuar siendo pequeño:

```python
from qfoil import (
    QFOIL,
    QFOILError,
    QFOILSyntaxError,
    QFOILValidationError,
    WitnessSearchResult,
)
```

Los componentes especializados permanecen en sus subpaquetes:

```python
from qfoil.language import ASTNode, QFOILParser, QFOILLatexRenderer
from qfoil.backends import ProbabilisticBackend
```

No se exportarán implementaciones internas de búsqueda, runtime probabilístico
ni adaptadores concretos opcionales desde `qfoil`.

## Compatibilidad

- `QFOIL.parse()`, `evaluate()`, `define_function()` y `grow_witness()`
  mantienen sus firmas actuales durante esta etapa.
- Las nuevas operaciones son aditivas.
- Los errores nuevos conservan compatibilidad con `ValueError` cuando se trate
  de entradas inválidas.
- PyJuice y Z3 continúan siendo extras opcionales.
- Importar `qfoil` no debe importar `torch`, `pyjuice`, `z3`, FastAPI ni KaTeX.

El proyecto sigue en versión `0.1.0`; cualquier ruptura futura debe registrarse
en notas de versión y reflejarse en pruebas de contrato.

## Plan de implementación

### Hito 1: contratos públicos

1. Crear `qfoil/errors.py`.
2. Crear `qfoil/types.py` con los alias estrictamente necesarios.
3. Documentar entradas, retornos y excepciones de los métodos existentes.
4. Sustituir errores genéricos en las fronteras públicas sin filtrar detalles
   internos innecesarios.
5. Actualizar las pruebas de exportaciones y compatibilidad.

### Hito 2: resultado de búsqueda

1. Crear `WitnessSearchResult` como `dataclass` inmutable.
2. Implementar `QFOIL.search_witness()`.
3. Verificar siempre el testigo antes de devolver `found`.
4. Implementar `to_dict()` y pruebas de serialización.
5. Mantener `grow_witness()` como API compacta compatible.

### Hito 3: LaTeX de expresiones QFOIL

1. Implementar `QFOILLatexRenderer` mediante un visitor exhaustivo del AST.
2. Definir reglas de precedencia y escape de identificadores.
3. Agregar `QFOIL.to_latex()`.
4. Cubrir todos los nodos y operadores con pruebas unitarias.
5. Verificar las cadenas generadas contra el subconjunto admitido por KaTeX;
   esta verificación no requiere incorporar KaTeX como dependencia Python.

### Hito 4: documentación y adopción

1. Añadir al README un recorrido de parseo, evaluación, búsqueda y LaTeX.
2. Documentar el significado heurístico de `not_found`.
3. Añadir ejemplos sin backend probabilístico y con
   `ProbabilisticBackend`.
4. Confirmar que los notebooks pueden migrar gradualmente sin romper las
   llamadas existentes.

## Estrategia de pruebas

### Pruebas de contrato

- exportaciones del paquete raíz;
- ausencia de importaciones de dependencias opcionales;
- compatibilidad de firmas existentes;
- jerarquía y mensajes de errores públicos;
- inmutabilidad y serialización de resultados.

### Pruebas del renderer

- un caso por cada tipo de nodo del AST;
- precedencia y paréntesis;
- identificadores con guion bajo;
- funciones anidadas;
- `SIGMA` anidado;
- tuplas con `None`;
- caracteres no permitidos y escape seguro;
- aceptación indistinta de texto y AST mediante `QFOIL.to_latex()`.

### Pruebas de búsqueda

- testigo encontrado y verificado;
- presupuesto agotado;
- semilla utilizada;
- opciones efectivas reflejadas en el resultado;
- error por binding faltante o longitud incompatible;
- equivalencia entre `grow_witness()` y `search_witness().witness`.

El comando de verificación principal es:

```bash
conda run -n pc pytest tests/
```

Las pruebas del núcleo y del renderer no deben requerir PyJuice, CUDA ni Z3.

## Criterios de terminación

La primera versión estable de esta API estará completa cuando:

1. los métodos actuales conserven compatibilidad y tengan contratos probados;
2. los consumidores puedan distinguir errores de sintaxis, validación y
   backend;
3. `search_witness()` devuelva un resultado verificado y estructurado;
4. cualquier expresión aceptada por el parser pueda convertirse a LaTeX o
   produzca un error público explícito;
5. importar el paquete no cargue dependencias opcionales;
6. README y pruebas muestren el flujo completo de uso desde Python.

## TODO bloqueado: lenguaje declarativo

Las siguientes decisiones y componentes quedan expresamente fuera de los
hitos anteriores hasta que se apruebe la sintaxis declarativa:

- [ ] Aprobar las palabras reservadas y la gramática superficial.
- [ ] Aprobar el papel exacto de `UNDER`, `FIND`, `WHERE`, `USING` y `REPORT`.
- [ ] Aprobar la sintaxis y precedencia de composición de relaciones.
- [ ] Aprobar la declaración de una o varias variables de búsqueda.
- [ ] Aprobar la sintaxis de propiedades como `candidate.SIZE`.
- [ ] Definir el contrato de registro de nociones o relaciones declarativas.
- [ ] Definir la inferencia de dominios bajo `AND`, `OR` y `NOT`.
- [ ] Crear el Query AST y el parser declarativo.
- [ ] Crear `PreparedQuery` y el compilador hacia QFOIL.
- [ ] Crear el executor declarativo y `QueryResult`.
- [ ] Definir el reporte predeterminado y la semántica final de `REPORT`.
- [ ] Crear el renderer LaTeX de la consulta declarativa compacta.
- [ ] Definir si la fórmula declarativa y la expandida se generan de forma
      inmediata, perezosa o configurable.
- [ ] Definir la API de búsqueda conjunta para múltiples variables.
- [ ] Crear `qfoil/queries/` y decidir sus exportaciones públicas.
- [ ] Actualizar `DECLARATIVE_MVP.md` una vez aprobada la sintaxis; hasta
      entonces ese documento se considera una propuesta de diseño, no un
      contrato de implementación.

FastAPI seguirá fuera de este proyecto incluso después de aprobar el lenguaje
declarativo. Un eventual servicio HTTP dependerá de esta API Python y no al
revés.
