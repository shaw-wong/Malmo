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
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="RewardDensity" type="{http://ProjectMalmo.microsoft.com}RewardDensityForBuildAndBreak"/>
 *         &lt;element name="AddQuitProducer" minOccurs="0">
 *           &lt;complexType>
 *             &lt;complexContent>
 *               &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *                 &lt;attribute name="description" type="{http://www.w3.org/2001/XMLSchema}string" default="" />
 *               &lt;/restriction>
 *             &lt;/complexContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *       &lt;/all>
 *       &lt;attGroup ref="{http://ProjectMalmo.microsoft.com}RewardProducerAttributes"/>
 *       &lt;attribute name="rewardScale" type="{http://www.w3.org/2001/XMLSchema}decimal" default="5.0" />
 *       &lt;attribute name="rewardDistribution" type="{http://www.w3.org/2001/XMLSchema}string" default="" />
 *       &lt;attribute name="rewardForCompletion" type="{http://www.w3.org/2001/XMLSchema}decimal" default="200.0" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {

})
@XmlRootElement(name = "RewardForStructureCopying")
public class RewardForStructureCopying {

    @XmlElement(name = "RewardDensity", required = true)
    protected RewardDensityForBuildAndBreak rewardDensity;
    @XmlElement(name = "AddQuitProducer")
    protected RewardForStructureCopying.AddQuitProducer addQuitProducer;
    @XmlAttribute(name = "rewardScale")
    protected BigDecimal rewardScale;
    @XmlAttribute(name = "rewardDistribution")
    protected String rewardDistribution;
    @XmlAttribute(name = "rewardForCompletion")
    protected BigDecimal rewardForCompletion;
    @XmlAttribute(name = "dimension")
    protected Integer dimension;

    /**
     * Gets the value of the rewardDensity property.
     * 
     * @return
     *     possible object is
     *     {@link RewardDensityForBuildAndBreak }
     *     
     */
    public RewardDensityForBuildAndBreak getRewardDensity() {
        return rewardDensity;
    }

    /**
     * Sets the value of the rewardDensity property.
     * 
     * @param value
     *     allowed object is
     *     {@link RewardDensityForBuildAndBreak }
     *     
     */
    public void setRewardDensity(RewardDensityForBuildAndBreak value) {
        this.rewardDensity = value;
    }

    /**
     * Gets the value of the addQuitProducer property.
     * 
     * @return
     *     possible object is
     *     {@link RewardForStructureCopying.AddQuitProducer }
     *     
     */
    public RewardForStructureCopying.AddQuitProducer getAddQuitProducer() {
        return addQuitProducer;
    }

    /**
     * Sets the value of the addQuitProducer property.
     * 
     * @param value
     *     allowed object is
     *     {@link RewardForStructureCopying.AddQuitProducer }
     *     
     */
    public void setAddQuitProducer(RewardForStructureCopying.AddQuitProducer value) {
        this.addQuitProducer = value;
    }

    /**
     * Gets the value of the rewardScale property.
     * 
     * @return
     *     possible object is
     *     {@link BigDecimal }
     *     
     */
    public BigDecimal getRewardScale() {
        if (rewardScale == null) {
            return new BigDecimal("5.0");
        } else {
            return rewardScale;
        }
    }

    /**
     * Sets the value of the rewardScale property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigDecimal }
     *     
     */
    public void setRewardScale(BigDecimal value) {
        this.rewardScale = value;
    }

    /**
     * Gets the value of the rewardDistribution property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getRewardDistribution() {
        if (rewardDistribution == null) {
            return "";
        } else {
            return rewardDistribution;
        }
    }

    /**
     * Sets the value of the rewardDistribution property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setRewardDistribution(String value) {
        this.rewardDistribution = value;
    }

    /**
     * Gets the value of the rewardForCompletion property.
     * 
     * @return
     *     possible object is
     *     {@link BigDecimal }
     *     
     */
    public BigDecimal getRewardForCompletion() {
        if (rewardForCompletion == null) {
            return new BigDecimal("200.0");
        } else {
            return rewardForCompletion;
        }
    }

    /**
     * Sets the value of the rewardForCompletion property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigDecimal }
     *     
     */
    public void setRewardForCompletion(BigDecimal value) {
        this.rewardForCompletion = value;
    }

    /**
     * Gets the value of the dimension property.
     * 
     * @return
     *     possible object is
     *     {@link Integer }
     *     
     */
    public int getDimension() {
        if (dimension == null) {
            return  0;
        } else {
            return dimension;
        }
    }

    /**
     * Sets the value of the dimension property.
     * 
     * @param value
     *     allowed object is
     *     {@link Integer }
     *     
     */
    public void setDimension(Integer value) {
        this.dimension = value;
    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;complexContent>
     *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
     *       &lt;attribute name="description" type="{http://www.w3.org/2001/XMLSchema}string" default="" />
     *     &lt;/restriction>
     *   &lt;/complexContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "")
    public static class AddQuitProducer {

        @XmlAttribute(name = "description")
        protected String description;

        /**
         * Gets the value of the description property.
         * 
         * @return
         *     possible object is
         *     {@link String }
         *     
         */
        public String getDescription() {
            if (description == null) {
                return "";
            } else {
                return description;
            }
        }

        /**
         * Sets the value of the description property.
         * 
         * @param value
         *     allowed object is
         *     {@link String }
         *     
         */
        public void setDescription(String value) {
            this.description = value;
        }

    }

}