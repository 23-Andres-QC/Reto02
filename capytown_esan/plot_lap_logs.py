#!/usr/bin/env python3
"""CapyTown plot_lap_logs - RC-2.

Script standalone SIN ROS (no necesita rclpy ni el robot corriendo) que lee
los CSV generados por lap_logger.py y genera los gráficos. Se puede correr
cuantas veces se quiera, después de la prueba, sin volver a mover el carrito.

Lee de `--dir` (por defecto /root):
  log_trayectoria.csv, log_velocidades.csv, log_error.csv,
  log_deteccion_amarillo.csv, log_deteccion_blanco.csv, log_slope.csv,
  log_giros.csv

Genera en el mismo directorio:
  velocidades_errores.png → velocidad lineal/angular y error lateral en el tiempo
  deteccion_colores.png   → posición de amarillo y blanco detectados en el tiempo
  recorrido.png           → trayectoria x,y del carrito
  slope_giros.png         → pendiente en el tiempo + marcas verticales de cada
                             giro cerrado detectado (para diagnosticar a qué
                             distancia/tiempo real se dispara cada giro, y si
                             hay esquinas que el carrito tomó pero que nunca
                             se contaron como "giro cerrado")

Uso:
  python3 plot_lap_logs.py
  python3 plot_lap_logs.py --dir /root/logs
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='/root', help='Carpeta con los CSV de lap_logger.py')
    args = parser.parse_args()
    d = args.dir

    traj  = _read_csv(f'{d}/log_trayectoria.csv')
    vel   = _read_csv(f'{d}/log_velocidades.csv')
    err   = _read_csv(f'{d}/log_error.csv')
    det_y = _read_csv(f'{d}/log_deteccion_amarillo.csv')
    det_w = _read_csv(f'{d}/log_deteccion_blanco.csv')
    slope = _read_csv(f'{d}/log_slope.csv')
    giros = _read_csv(f'{d}/log_giros.csv')

    print(f'Leídos: {len(traj)} puntos trayectoria, {len(vel)} velocidades, '
          f'{len(err)} errores, {len(det_y)} amarillo, {len(det_w)} blanco, '
          f'{len(slope)} slope, {len(giros)} giros completados.')

    # ---- velocidades_errores.png ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot([float(r['t']) for r in vel], [float(r['linear']) for r in vel], '-b', label='Velocidad lineal (m/s)')
    ax1.plot([float(r['t']) for r in vel], [float(r['angular']) for r in vel], '-g', label='Velocidad angular (rad/s)')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_ylabel('velocidad')
    ax1.set_title(f'Velocidades y error — {len(giros)} giros cerrados')
    ax1.legend(); ax1.grid(True)

    ax2.plot([float(r['t']) for r in err], [float(r['error_cm']) for r in err], '-r', label='Error lateral (cm)')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('tiempo (s)'); ax2.set_ylabel('error (cm)')
    ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    fig.savefig(f'{d}/velocidades_errores.png')
    plt.close(fig)

    # ---- deteccion_colores.png ----
    fig = plt.figure(figsize=(10, 4))
    plt.plot([float(r['t']) for r in det_y], [float(r['pos_cm']) for r in det_y], '-', color='gold', label='Amarillo (cm desde el centro)')
    plt.plot([float(r['t']) for r in det_w], [float(r['pos_cm']) for r in det_w], '-', color='gray', label='Blanco (cm desde el centro)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('tiempo (s)'); plt.ylabel('posición (cm)')
    plt.title('Detección de carril — amarillo y blanco')
    plt.legend(); plt.grid(True)
    fig.tight_layout()
    fig.savefig(f'{d}/deteccion_colores.png')
    plt.close(fig)

    # ---- recorrido.png ----
    xs = [float(r['x']) for r in traj]
    ys = [float(r['y']) for r in traj]
    fig = plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, '-b', linewidth=1.5, label='Recorrido')
    if xs:
        plt.plot(xs[0], ys[0], 'go', markersize=10, label='Inicio')
        plt.plot(xs[-1], ys[-1], 'rx', markersize=10, markeredgewidth=2, label='Fin')
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title(f'Recorrido del carrito — {len(giros)} giros cerrados')
    plt.axis('equal'); plt.legend(); plt.grid(True)
    fig.tight_layout()
    fig.savefig(f'{d}/recorrido.png')
    plt.close(fig)

    # ---- slope_giros.png (diagnóstico) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    st = [float(r['t']) for r in slope]
    sv = [float(r['slope_cm']) for r in slope]
    sturn = [int(r['in_sharp_turn']) * 20 for r in slope]   # escalado solo para visualizar el estado
    ax.plot(st, sv, '-b', linewidth=0.8, label='Pendiente (slope, cm)')
    ax.plot(st, sturn, '-', color='orange', linewidth=1.2, label='in_sharp_turn (escalado x20 para verlo)')
    for r in giros:
        ax.axvline(float(r['t']), color='red', linestyle='--', linewidth=1)
        ax.text(float(r['t']), max(sv, default=0), f"#{r['giro_num']}", color='red', fontsize=8, rotation=90, va='top')
    ax.axhline(13, color='gray', linestyle=':', label='umbral esquina (13cm)')
    ax.axhline(-13, color='gray', linestyle=':')
    ax.axhline(4, color='lightgray', linestyle=':', label='umbral anticipación (4cm)')
    ax.axhline(-4, color='lightgray', linestyle=':')
    ax.set_xlabel('tiempo (s)'); ax.set_ylabel('pendiente (cm) / estado')
    ax.set_title('Pendiente vs tiempo, con cada giro cerrado marcado (línea roja)')
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'{d}/slope_giros.png')
    plt.close(fig)

    print(f'Gráficos guardados en {d}/: velocidades_errores.png, deteccion_colores.png, '
          f'recorrido.png, slope_giros.png')


if __name__ == '__main__':
    main()
