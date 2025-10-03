import numpy as np
import pygad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ========== CONFIGURACIÓN DEL HORARIO ==========
BLOQUES = [
    "07:20-08:05", "08:05-08:50", "08:50-09:35", "09:35-10:20",
    "10:40-11:25", "11:25-12:10", "12:10-12:55",
    "13:25-14:10", "14:10-14:55"
]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes"]
NUM_BLOQUES = 9
NUM_DIAS = 5

@dataclass
class Docente:
    id: int  # Cambio a int para indexación en NumPy
    nombre: str
    tipo: str
    horas_requeridas: int
    disponibilidad: np.ndarray  # Shape: (5 días, 9 bloques)

@dataclass
class Clase:
    id: int
    docente_id: int
    curso: str
    grado: int
    seccion: str
    horas_semanales: int
    aula: str  # Aula fija por grado-sección

class GeneticScheduler:
    """
    Codificación del Cromosoma:
    - Cromosoma: vector 1D de longitud = suma(horas_semanales de todas las clases)
    - Cada gen representa: (día, bloque) donde se asigna esa hora de clase
    - Gen = día * NUM_BLOQUES + bloque  (valor entre 0 y 44)
    
    Ejemplo: gen=15 significa día=1 (martes), bloque=6
    """
    
    def __init__(self, docentes: List[Docente], clases: List[Clase], 
                 grados_secciones: List[str], num_generaciones: int = 100):
        self.docentes = {d.id: d for d in docentes}
        self.clases = clases
        self.grados_secciones = grados_secciones
        self.num_generaciones = num_generaciones
        
        # Pesos de penalización
        self.w_huecos = 10.0
        self.w_conflictos = 1000.0
        self.w_horas = 500.0
        self.w_disponibilidad = 800.0
        
        # Calcular longitud del cromosoma
        self.longitud_cromosoma = sum(c.horas_semanales for c in clases)
        
        # Crear mapeo clase_id -> índices en cromosoma
        self.clase_indices = {}
        idx = 0
        for clase in clases:
            self.clase_indices[clase.id] = list(range(idx, idx + clase.horas_semanales))
            idx += clase.horas_semanales
        
        print(f"Cromosoma inicializado: {self.longitud_cromosoma} genes")
        print(f"Configuración: {len(clases)} clases, {len(docentes)} docentes")
    
    def decodificar_gen(self, gen_valor: int) -> Tuple[int, int]:
        """Convierte valor del gen a (día, bloque)"""
        dia = gen_valor // NUM_BLOQUES
        bloque = gen_valor % NUM_BLOQUES
        return int(dia), int(bloque)
    
    def codificar_gen(self, dia: int, bloque: int) -> int:
        """Convierte (día, bloque) a valor del gen"""
        return dia * NUM_BLOQUES + bloque
    
    def fitness_function(self, ga_instance, solution, solution_idx):
        """
        Función de fitness para PyGAD (a MAXIMIZAR)
        Retornamos el negativo de la penalización total
        """
        penalizacion_total = (
            self.w_huecos * self._calc_huecos(solution) +
            self.w_conflictos * self._calc_conflictos(solution) +
            self.w_horas * self._calc_horas(solution) +
            self.w_disponibilidad * self._calc_disponibilidad(solution)
        )
        
        # PyGAD maximiza, así que retornamos el negativo
        return -penalizacion_total
    
    def _calc_huecos(self, cromosoma: np.ndarray) -> float:
        """
        P_huecos = Σ_docentes Σ_días (bloque_max - bloque_min + 1 - bloques_ocupados)
        """
        penalizacion = 0.0
        
        for docente_id in self.docentes.keys():
            # Obtener todas las asignaciones de este docente
            asignaciones = []
            for clase in self.clases:
                if clase.docente_id == docente_id:
                    indices = self.clase_indices[clase.id]
                    for idx in indices:
                        dia, bloque = self.decodificar_gen(int(cromosoma[idx]))
                        if 0 <= dia < NUM_DIAS:  # Validación
                            asignaciones.append((dia, bloque))
            
            # Agrupar por día y calcular huecos
            for dia in range(NUM_DIAS):
                bloques_dia = sorted([b for d, b in asignaciones if d == dia])
                
                if len(bloques_dia) > 1:
                    rango = bloques_dia[-1] - bloques_dia[0] + 1
                    huecos = rango - len(bloques_dia)
                    penalizacion += huecos
        
        return penalizacion
    
    def _calc_conflictos(self, cromosoma: np.ndarray) -> float:
        """
        P_conflictos = Σ max(0, N_aula_dia_bloque - 1)
        Detecta múltiples docentes en misma aula/horario
        """
        penalizacion = 0.0
        ocupacion = {}  # (aula, dia, bloque) -> count
        
        for clase in self.clases:
            indices = self.clase_indices[clase.id]
            for idx in indices:
                dia, bloque = self.decodificar_gen(int(cromosoma[idx]))
                if 0 <= dia < NUM_DIAS and 0 <= bloque < NUM_BLOQUES:
                    key = (clase.aula, dia, bloque)
                    ocupacion[key] = ocupacion.get(key, 0) + 1
        
        for count in ocupacion.values():
            if count > 1:
                penalizacion += (count - 1) * 10  # Penalización multiplicada
        
        return penalizacion
    
    def _calc_horas(self, cromosoma: np.ndarray) -> float:
        """
        P_horas = Σ_docentes |H_asignadas - H_requeridas|
        """
        penalizacion = 0.0
        horas_asignadas = {d_id: 0 for d_id in self.docentes.keys()}
        
        for clase in self.clases:
            indices = self.clase_indices[clase.id]
            horas_validas = 0
            for idx in indices:
                dia, bloque = self.decodificar_gen(int(cromosoma[idx]))
                if 0 <= dia < NUM_DIAS and 0 <= bloque < NUM_BLOQUES:
                    horas_validas += 1
            horas_asignadas[clase.docente_id] += horas_validas
        
        for docente_id, docente in self.docentes.items():
            diferencia = abs(horas_asignadas[docente_id] - docente.horas_requeridas)
            penalizacion += diferencia
        
        return penalizacion
    
    def _calc_disponibilidad(self, cromosoma: np.ndarray) -> float:
        """
        P_disponibilidad = Σ violaciones de disponibilidad del docente
        """
        penalizacion = 0.0
        
        for clase in self.clases:
            docente = self.docentes[clase.docente_id]
            indices = self.clase_indices[clase.id]
            
            for idx in indices:
                dia, bloque = self.decodificar_gen(int(cromosoma[idx]))
                if 0 <= dia < NUM_DIAS and 0 <= bloque < NUM_BLOQUES:
                    if not docente.disponibilidad[dia, bloque]:
                        penalizacion += 1
        
        return penalizacion
    
    def crear_poblacion_inicial(self, num_soluciones: int) -> np.ndarray:
        """
        Genera población inicial con heurística semi-aleatoria
        """
        poblacion = []
        
        for _ in range(num_soluciones):
            cromosoma = np.zeros(self.longitud_cromosoma, dtype=int)
            
            for clase in self.clases:
                docente = self.docentes[clase.docente_id]
                indices = self.clase_indices[clase.id]
                
                # Obtener slots disponibles del docente
                slots_disponibles = []
                for dia in range(NUM_DIAS):
                    for bloque in range(NUM_BLOQUES):
                        if docente.disponibilidad[dia, bloque]:
                            slots_disponibles.append(self.codificar_gen(dia, bloque))
                
                # Asignar aleatoriamente pero respetando disponibilidad
                if len(slots_disponibles) >= len(indices):
                    asignados = np.random.choice(slots_disponibles, size=len(indices), replace=False)
                    cromosoma[indices] = asignados
                else:
                    # Si no hay suficientes slots, usar aleatorios (será penalizado)
                    cromosoma[indices] = np.random.randint(0, NUM_DIAS * NUM_BLOQUES, size=len(indices))
            
            poblacion.append(cromosoma)
        
        return np.array(poblacion)
    
    def ejecutar(self):
        """Ejecuta el algoritmo genético con PyGAD"""
        
        # Definir rangos de genes (0 a 44: 5 días * 9 bloques - 1)
        gene_space = list(range(NUM_DIAS * NUM_BLOQUES))
        
        # Configurar PyGAD
        ga_instance = pygad.GA(
            num_generations=self.num_generaciones,
            num_parents_mating=10,
            fitness_func=self.fitness_function,
            sol_per_pop=50,
            num_genes=self.longitud_cromosoma,
            gene_space=gene_space,
            gene_type=int,
            parent_selection_type="tournament",
            K_tournament=3,
            crossover_type="single_point",
            crossover_probability=0.8,
            mutation_type="random",
            mutation_probability=0.15,
            keep_elitism=5,
            initial_population=self.crear_poblacion_inicial(50),
            suppress_warnings=True
        )
        
        # Ejecutar
        print("\n🧬 Iniciando optimización genética...")
        ga_instance.run()
        
        # Obtener mejor solución
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        print(f"\n✅ Optimización completada!")
        print(f"Fitness de mejor solución: {solution_fitness:.2f}")
        print(f"Penalización total: {-solution_fitness:.2f}")
        
        # Desglose de penalizaciones
        print("\n📊 Desglose de penalizaciones:")
        print(f"  - Huecos: {self._calc_huecos(solution):.0f}")
        print(f"  - Conflictos de aula: {self._calc_conflictos(solution):.0f}")
        print(f"  - Horas incorrectas: {self._calc_horas(solution):.0f}")
        print(f"  - Disponibilidad violada: {self._calc_disponibilidad(solution):.0f}")
        
        # Gráfica de evolución
        ga_instance.plot_fitness()
        
        return solution, ga_instance
    
    def generar_horarios_docentes(self, solucion: np.ndarray) -> Dict[int, pd.DataFrame]:
        """
        Genera DataFrames con el horario de cada docente
        Formato: Filas = Bloques horarios, Columnas = Días
        """
        horarios = {}
        
        for docente_id, docente in self.docentes.items():
            # Crear DataFrame vacío
            horario_df = pd.DataFrame(
                index=BLOQUES,
                columns=DIAS
            )
            horario_df = horario_df.fillna("")
            
            # Llenar con las asignaciones
            for clase in self.clases:
                if clase.docente_id == docente_id:
                    indices = self.clase_indices[clase.id]
                    for idx in indices:
                        dia, bloque = self.decodificar_gen(int(solucion[idx]))
                        if 0 <= dia < NUM_DIAS and 0 <= bloque < NUM_BLOQUES:
                            dia_nombre = DIAS[dia]
                            bloque_horario = BLOQUES[bloque]
                            
                            # Formato: "Curso - Grado Sección (Aula)"
                            info = f"{clase.curso}\n{clase.grado}{clase.seccion} ({clase.aula})"
                            
                            # Si ya hay algo en ese slot, concatenar (error a detectar)
                            celda_actual = horario_df.loc[bloque_horario, dia_nombre]
                            if celda_actual and str(celda_actual).strip():
                                horario_df.loc[bloque_horario, dia_nombre] = str(celda_actual) + f"\n⚠️ {info}"
                            else:
                                horario_df.loc[bloque_horario, dia_nombre] = info
            
            horarios[docente_id] = horario_df
        
        return horarios
    
    def visualizar_horarios(self, solucion: np.ndarray, guardar: bool = True):
        """
        Genera visualización gráfica de los horarios por docente
        """
        horarios = self.generar_horarios_docentes(solucion)
        
        for docente_id, horario_df in horarios.items():
            docente = self.docentes[docente_id]
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Preparar matriz para visualización
            matriz_ocupacion = np.zeros((NUM_BLOQUES, NUM_DIAS))
            
            for i, bloque in enumerate(BLOQUES):
                for j, dia in enumerate(DIAS):
                    celda = horario_df.loc[bloque, dia]
                    if celda and str(celda).strip():
                        matriz_ocupacion[i, j] = 1
            
            # Crear cuadrícula manualmente con matplotlib
            for i in range(NUM_BLOQUES):
                for j in range(NUM_DIAS):
                    # Color según ocupación
                    if matriz_ocupacion[i, j] == 1:
                        color = '#4CAF50'  # Verde para ocupado
                    else:
                        color = '#f0f0f0'  # Gris claro para vacío
                    
                    # Dibujar rectángulo
                    rect = Rectangle((j, NUM_BLOQUES - i - 1), 1, 1, 
                                    facecolor=color, edgecolor='white', linewidth=2)
                    ax.add_patch(rect)
                    
                    # Agregar texto
                    celda_texto = horario_df.loc[BLOQUES[i], DIAS[j]]
                    if celda_texto and str(celda_texto).strip():
                        ax.text(j + 0.5, NUM_BLOQUES - i - 0.5, str(celda_texto),
                               ha='center', va='center', fontsize=9, 
                               wrap=True, color='white', weight='bold')
            
            # Líneas para recreo y almuerzo
            ax.axhline(y=NUM_BLOQUES - 4, color='#FF9800', linewidth=4, 
                      label='Recreo (10:20-10:40)', zorder=10)
            ax.axhline(y=NUM_BLOQUES - 7, color='#FF5722', linewidth=4, 
                      label='Almuerzo (12:55-13:25)', zorder=10)
            
            # Configurar ejes
            ax.set_xlim(0, NUM_DIAS)
            ax.set_ylim(0, NUM_BLOQUES)
            ax.set_xticks(np.arange(NUM_DIAS) + 0.5)
            ax.set_xticklabels([d.upper() for d in DIAS], fontsize=11, weight='bold')
            ax.set_yticks(np.arange(NUM_BLOQUES) + 0.5)
            ax.set_yticklabels(BLOQUES[::-1], fontsize=10)
            
            # Título
            ax.set_title(
                f'HORARIO: {docente.nombre} ({docente.tipo.upper()})\n'
                f'Horas Asignadas: {self._contar_horas_docente(solucion, docente_id)} / '
                f'{docente.horas_requeridas}',
                fontsize=16,
                fontweight='bold',
                pad=20
            )
            
            ax.set_xlabel('DÍA', fontsize=12, fontweight='bold')
            ax.set_ylabel('HORARIO', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            
            # Quitar bordes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, bottom=False)
            
            plt.tight_layout()
            
            if guardar:
                nombre_archivo = f'horario_docente_{docente_id}_{docente.nombre.replace(" ", "_")}.png'
                plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
                print(f"✅ Guardado: {nombre_archivo}")
            
            plt.show()
            plt.close()
    
    def _contar_horas_docente(self, solucion: np.ndarray, docente_id: int) -> int:
        """Cuenta las horas asignadas a un docente"""
        total = 0
        for clase in self.clases:
            if clase.docente_id == docente_id:
                indices = self.clase_indices[clase.id]
                for idx in indices:
                    dia, bloque = self.decodificar_gen(int(solucion[idx]))
                    if 0 <= dia < NUM_DIAS and 0 <= bloque < NUM_BLOQUES:
                        total += 1
        return total
    
    def exportar_excel(self, solucion: np.ndarray, nombre_archivo: str = "horarios_docentes.xlsx"):
        """
        Exporta todos los horarios a un archivo Excel con una hoja por docente
        """
        horarios = self.generar_horarios_docentes(solucion)
        
        with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
            for docente_id, horario_df in horarios.items():
                docente = self.docentes[docente_id]
                nombre_hoja = f"{docente.nombre[:25]}"  # Excel limita a 31 caracteres
                
                # Agregar información del docente
                info_df = pd.DataFrame({
                    'Información': [
                        f'Docente: {docente.nombre}',
                        f'Tipo: {docente.tipo}',
                        f'Horas requeridas: {docente.horas_requeridas}',
                        f'Horas asignadas: {self._contar_horas_docente(solucion, docente_id)}',
                        ''
                    ]
                })
                
                info_df.to_excel(writer, sheet_name=nombre_hoja, index=False, startrow=0)
                horario_df.to_excel(writer, sheet_name=nombre_hoja, startrow=6)
        
        print(f"📊 Horarios exportados a: {nombre_archivo}")
    
    def imprimir_resumen(self, solucion: np.ndarray):
        """Imprime un resumen en consola de todos los horarios"""
        horarios = self.generar_horarios_docentes(solucion)
        
        print("\n" + "="*100)
        print("📚 HORARIOS GENERADOS POR DOCENTE")
        print("="*100)
        
        for docente_id, horario_df in horarios.items():
            docente = self.docentes[docente_id]
            horas_asignadas = self._contar_horas_docente(solucion, docente_id)
            
            print(f"\n{'='*100}")
            print(f"👤 DOCENTE: {docente.nombre.upper()} ({docente.tipo})")
            print(f"⏱️  HORAS: {horas_asignadas}/{docente.horas_requeridas}")
            print(f"{'='*100}")
            print(horario_df.to_string())
            print()


# ========== EJEMPLO DE USO ==========
if __name__ == "__main__":
    # Crear matriz de disponibilidad (5 días x 9 bloques)
    disponibilidad_completa = np.ones((NUM_DIAS, NUM_BLOQUES), dtype=bool)
    
    disponibilidad_parcial = np.ones((NUM_DIAS, NUM_BLOQUES), dtype=bool)
    disponibilidad_parcial[0, 5:] = False  # Lunes tarde no disponible
    disponibilidad_parcial[3, 5:] = False  # Jueves tarde no disponible
    
    # Crear docentes
    docentes = [
        Docente(
            id=0,
            nombre="Edwin García",
            tipo="nombrado",
            horas_requeridas=12,  # Ajustado para prueba
            disponibilidad=disponibilidad_completa
        ),
        Docente(
            id=1,
            nombre="Carlos Mendoza",
            tipo="contratado",
            horas_requeridas=10,
            disponibilidad=disponibilidad_parcial
        ),
        Docente(
            id=2,
            nombre="María López",
            tipo="nombrado",
            horas_requeridas=8,
            disponibilidad=disponibilidad_completa
        )
    ]
    
    # Crear clases (simplificado para 3 grados)
    clases = [
        Clase(0, 0, "Matemática", 3, "A", 4, "AULA_3A"),
        Clase(1, 0, "Matemática", 3, "B", 4, "AULA_3B"),
        Clase(2, 1, "Computación", 3, "A", 3, "AULA_3A"),
        Clase(3, 1, "Computación", 4, "A", 3, "AULA_4A"),
        Clase(4, 2, "Comunicación", 3, "B", 4, "AULA_3B"),
        Clase(5, 0, "Física", 4, "A", 4, "AULA_4A"),
        Clase(6, 2, "Historia", 3, "A", 4, "AULA_3A"),
    ]
    
    # Ejecutar optimización
    scheduler = GeneticScheduler(
        docentes=docentes,
        clases=clases,
        grados_secciones=["3A", "3B", "4A"],
        num_generaciones=150
    )
    
    mejor_solucion, ga_instance = scheduler.ejecutar()
    
    # Mostrar resumen en consola
    scheduler.imprimir_resumen(mejor_solucion)
    
    # Exportar a Excel
   # scheduler.exportar_excel(mejor_solucion, "horarios_docentes_optimizados.xlsx")
    
    # Visualizar gráficamente
    scheduler.visualizar_horarios(mejor_solucion, guardar=True)