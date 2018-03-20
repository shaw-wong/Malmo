//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2018.03.06 at 04:37:50 PM AEDT 
//


package com.microsoft.Malmo.Schemas;

import java.math.BigDecimal;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for BlockSpecWithRewardAndBehaviour complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="BlockSpecWithRewardAndBehaviour">
 *   &lt;complexContent>
 *     &lt;extension base="{http://ProjectMalmo.microsoft.com}BlockSpec">
 *       &lt;attribute name="reward" use="required" type="{http://www.w3.org/2001/XMLSchema}decimal" />
 *       &lt;attribute name="distribution" type="{http://www.w3.org/2001/XMLSchema}string" default="" />
 *       &lt;attribute name="behaviour" type="{http://ProjectMalmo.microsoft.com}Behaviour" default="oncePerBlock" />
 *       &lt;attribute name="cooldownInMs" type="{http://www.w3.org/2001/XMLSchema}decimal" default="1000" />
 *     &lt;/extension>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "BlockSpecWithRewardAndBehaviour")
public class BlockSpecWithRewardAndBehaviour
    extends BlockSpec
{

    @XmlAttribute(name = "reward", required = true)
    protected BigDecimal reward;
    @XmlAttribute(name = "distribution")
    protected String distribution;
    @XmlAttribute(name = "behaviour")
    protected Behaviour behaviour;
    @XmlAttribute(name = "cooldownInMs")
    protected BigDecimal cooldownInMs;

    /**
     * Gets the value of the reward property.
     * 
     * @return
     *     possible object is
     *     {@link BigDecimal }
     *     
     */
    public BigDecimal getReward() {
        return reward;
    }

    /**
     * Sets the value of the reward property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigDecimal }
     *     
     */
    public void setReward(BigDecimal value) {
        this.reward = value;
    }

    /**
     * Gets the value of the distribution property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDistribution() {
        if (distribution == null) {
            return "";
        } else {
            return distribution;
        }
    }

    /**
     * Sets the value of the distribution property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDistribution(String value) {
        this.distribution = value;
    }

    /**
     * Gets the value of the behaviour property.
     * 
     * @return
     *     possible object is
     *     {@link Behaviour }
     *     
     */
    public Behaviour getBehaviour() {
        if (behaviour == null) {
            return Behaviour.ONCE_PER_BLOCK;
        } else {
            return behaviour;
        }
    }

    /**
     * Sets the value of the behaviour property.
     * 
     * @param value
     *     allowed object is
     *     {@link Behaviour }
     *     
     */
    public void setBehaviour(Behaviour value) {
        this.behaviour = value;
    }

    /**
     * Gets the value of the cooldownInMs property.
     * 
     * @return
     *     possible object is
     *     {@link BigDecimal }
     *     
     */
    public BigDecimal getCooldownInMs() {
        if (cooldownInMs == null) {
            return new BigDecimal("1000");
        } else {
            return cooldownInMs;
        }
    }

    /**
     * Sets the value of the cooldownInMs property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigDecimal }
     *     
     */
    public void setCooldownInMs(BigDecimal value) {
        this.cooldownInMs = value;
    }

}
