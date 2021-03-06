#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../utilities/array.h"

#include "struct_particles.h"
#include "particles.h"


void Particles_Alloc(struct Particles **sources_addr, int length)
{
    *sources_addr = malloc(sizeof(struct Particles));
    struct Particles *sources = *sources_addr;

	sources->num = length;
    sources->x = NULL;
    sources->y = NULL;
    sources->z = NULL;
    sources->q = NULL;
    
    if (sources->num > 0) {
        make_vector(sources->x, sources->num);
        make_vector(sources->y, sources->num);
        make_vector(sources->z, sources->num);
        make_vector(sources->q, sources->num);
    }

    return;
}



void Particles_Free(struct Particles **sources_addr)
{
    struct Particles *sources = *sources_addr;

    if (sources != NULL) {
	    if (sources->x != NULL) free_vector(sources->x);
	    if (sources->y != NULL) free_vector(sources->y);
	    if (sources->z != NULL) free_vector(sources->z);
	    if (sources->q != NULL) free_vector(sources->q);
        free(sources);
    }
    
    sources = NULL;

    return;
}



void Particles_Sources_Reorder(struct Particles *sources)
{
    int numpars = sources->num;
    int *reorder = sources->order;

    double *temp_x;
    make_vector(temp_x, numpars);
    for (int i = 0; i < numpars; i++) temp_x[i] = sources->x[i];
    for (int i = 0; i < numpars; i++) sources->x[reorder[i]-1] = temp_x[i];
    free_vector(temp_x);

    double *temp_y;
    make_vector(temp_y, numpars);
    for (int i = 0; i < numpars; i++) temp_y[i] = sources->y[i];
    for (int i = 0; i < numpars; i++) sources->y[reorder[i]-1] = temp_y[i];
    free_vector(temp_y);

    double *temp_z;
    make_vector(temp_z, numpars);
    for (int i = 0; i < numpars; i++) temp_z[i] = sources->z[i];
    for (int i = 0; i < numpars; i++) sources->z[reorder[i]-1] = temp_z[i];
    free_vector(temp_z);

    double *temp_q;
    make_vector(temp_q, numpars);
    for (int i = 0; i < numpars; i++) temp_q[i] = sources->q[i];
    for (int i = 0; i < numpars; i++) sources->q[reorder[i]-1] = temp_q[i];
    free_vector(temp_q);

    return;
}



void Particles_ConstructOrder(struct Particles *particles)
{
    make_vector(particles->order, particles->num);
    for (int i = 0; i < particles->num; i++) particles->order[i] = i+1;
    
    return;
}



void Particles_FreeOrder(struct Particles *particles)
{
    if (particles->order != NULL) free_vector(particles->order);
    
    return;
}
