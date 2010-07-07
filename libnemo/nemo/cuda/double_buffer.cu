/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/* Data structures which are used for communication between different
 * partitions, need to be double buffered so as to avoid race conditions.
 * These functions return the double buffer index (0 or 1) for the given cycle,
 * for either the read or write part of the buffer */

__device__
unsigned
readBuffer(unsigned cycle)
{
    return (cycle & 0x1) ^ 0x1;
}


__device__
unsigned
writeBuffer(unsigned cycle)
{
    return cycle & 0x1;
}
